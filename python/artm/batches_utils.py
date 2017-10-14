# Copyright 2017, Additive Regularization of Topic Models.

import os
import glob
import uuid
import shutil
import numpy as np

from six import iteritems, string_types
from six.moves import range, zip

from . import wrapper
from .wrapper import constants as const
from .wrapper import messages_pb2 as messages

from .dictionary import Dictionary


__all__ = [
    'BatchVectorizer'
]

GLOB_EPS = 1e-37


class Batch(object):
    def __init__(self, filename):
        self._filename = filename

    def __repr__(self):
        return 'Batch({0})'.format(self._filename)

    @property
    def filename(self):
        return self._filename


class BatchVectorizer(object):
    def __init__(self, batches=None, collection_name=None, data_path='', data_format='batches',
                 target_folder=None, batch_size=1000, batch_name_type='code', data_weight=1.0, n_wd=None,
                 vocabulary=None, gather_dictionary=True, class_ids=None, process_in_memory_model=None):
        """
        :param str collection_name: the name of text collection (required if data_format == 'bow_uci')
        :param str data_path: 1) if data_format == 'bow_uci' => folder containing\
                                 'docword.collection_name.txt' and vocab.collection_name.txt files;\
                              2) if data_format == 'vowpal_wabbit' => file in Vowpal Wabbit format;\
                              3) if data_format == 'bow_n_wd' => useless parameter\
                              4) if data_format == 'batches' => folder containing batches
        :param str data_format: the type of input data:\
                              1) 'bow_uci' --- Bag-Of-Words in UCI format;\
                              2) 'vowpal_wabbit' --- Vowpal Wabbit format;\
                              3  'bow_n_wd' --- result of CountVectorizer or similar tool;\
                              4) 'batches' --- the BigARTM data format
        :param int batch_size: number of documents to be stored in each batch
        :param str target_folder: full path to folder for future batches storing;\
                                  if not set, no batches will be produced for further work
        :param batches: if process_in_memory_model is None -> list with non-full file names of\
                              batches (necessary parameters are batches + data_path +\
                              data_fromat=='batches' in this case)\
                        else -> list of batches (messages.Batch objects), loaded in memory
        :type batches: list of str
        :param str batch_name_type: name batches in natural order ('code') or using random guids (guid)
        :param float data_weight: weight for a group of batches from data_path;\
                              it can be a list of floats, then data_path (and\
                              target_folder if not data_format == 'batches')\
                              should also be lists; one weight corresponds to\
                              one path from the data_path list;
        :param array n_wd: matrix with n_wd counters
        :param dict vocabulary: dict with vocabulary, key - index of n_wd, value - token
        :param bool gather_dictionary: create or not the default dictionary in vectorizer;\
                                       if data_format == 'bow_n_wd' - automatically set to True;\
                                       and if data_format == 'batches' or data_weight is list -\
                                       automatically set to False
        :param class_ids: list of class_ids or single class_id to parse and include in batches
        :type class_ids: list of str or str
        :param artm.ARTM process_in_memory_model: ARTM instance that will use this vectorizer, is\
                                                  required when one needs processing of batches from\
                                                  disk in RAM (only if data_format == 'batches').\
                                                  NOTE: makes vectorizer model specific.
        """
        self._remove_batches = False
        self._process_in_memory = data_format == 'batches' and process_in_memory_model is not None
        if not self._process_in_memory and process_in_memory_model is not None:
            raise IOError("Correct configuration for in memory processing: data_format =="
                          "'batches' + process_in_memory_model != None")

        self._model = process_in_memory_model
        if data_format == 'bow_n_wd' or data_format == 'vowpal_wabbit' or data_format == 'bow_uci':
            self._remove_batches = target_folder is None
        elif data_format == 'batches':
            self._remove_batches = False

        self._target_folder = target_folder
        if self._remove_batches:
            self._target_folder = os.path.join(data_path, format(uuid.uuid1().urn).replace(':', ''))

        self._batches_list = []
        self._weights = []
        self._data_path = data_path
        self._batch_size = batch_size

        self._dictionary = None
        if gather_dictionary and not isinstance(data_weight, list) and data_format != 'batches':
            self._dictionary = Dictionary()

        if data_format == 'bow_n_wd':
            self._parse_n_wd(data_weight=1.0, n_wd=n_wd, vocab=vocabulary)
        elif data_format == 'batches':
            self._parse_batches(data_weight=data_weight, batches=batches)
        elif data_format == 'vowpal_wabbit':
            self._parse_uci_or_vw(data_weight=data_weight, format='vw', class_ids=class_ids)
        elif data_format == 'bow_uci':
            self._parse_uci_or_vw(data_weight=data_weight,
                                  format='uci',
                                  col_name=collection_name,
                                  batch_name_type=batch_name_type,
                                  class_ids=class_ids)
        else:
            raise IOError('Unknown data format')

        self._data_path = data_path if data_format == 'batches' else self._target_folder

    def __dispose(self):
        if self._process_in_memory:
            for batch in self._batches_list:
                self._model.master.remove_batch(batch)
        self._process_in_memory = False

        if self._remove_batches:
            shutil.rmtree(self._target_folder)
        self._remove_batches = False

    def __exit__(self, exc_type, exc_value, traceback):
        self.__dispose()

    def __del__(self):
        self.__dispose()

    def _populate_data(self, data_weight, is_batches=False):
        """
        This method create lists of input parameters for processing.
        It converts input scalars to lists if it is necessary.
        """
        if isinstance(self._data_path, list):
            data_paths = self._data_path
            if len(self._data_path) != len(data_weight):
                raise IOError('Lists for data_path and data_weight should have the same length')
            data_weights = data_weight
            if is_batches:
                target_folders = ['' for p in data_paths]
            else:
                if len(self._data_path) != len(self._target_folder):
                    raise IOError('Lists for data_path and target_folder should have same length')
                target_folders = self._target_folder
        else:
            if isinstance(data_weight, list) or isinstance(self._target_folder, list):
                raise IOError('data_path should be also a list for multiple weights or folders')
            data_paths = [self._data_path]
            data_weights = [data_weight]
            target_folders = [self._target_folder]

        return data_paths, data_weights, target_folders

    def _parse_uci_or_vw(self, data_weight=None, format=None, col_name=None, batch_name_type=None, class_ids=None):
        data_paths, data_weights, target_folders = self._populate_data(data_weight)
        for (data_p, data_w, target_f) in zip(data_paths, data_weights, target_folders):
            parser_config = messages.CollectionParserConfig()

            parser_config.num_items_per_batch = self._batch_size
            parser_config.target_folder = target_f

            if class_ids is not None:
                if isinstance(class_ids, string_types):
                    class_ids = [class_ids]
                for class_id in class_ids:
                    parser_config.class_id.append(class_id)

            if format == 'uci':
                parser_config.docword_file_path = os.path.join(data_p, 'docword.{0}.txt'.format(col_name))
                parser_config.vocab_file_path = os.path.join(data_p, 'vocab.{0}.txt'.format(col_name))
                parser_config.format = const.CollectionParserConfig_CollectionFormat_BagOfWordsUci
            elif format == 'vw':
                parser_config.docword_file_path = data_p
                parser_config.format = const.CollectionParserConfig_CollectionFormat_VowpalWabbit

            parser_config.name_type = const.CollectionParserConfig_BatchNameType_Code
            if batch_name_type == 'guid':
                parser_config.name_type = const.CollectionParserConfig_BatchNameType_Guid

            lib = wrapper.LibArtm()
            lib.ArtmParseCollection(parser_config)
            batch_filenames = glob.glob(os.path.join(target_f, '*.batch'))
            self._batches_list += [Batch(filename) for filename in batch_filenames]
            self._weights += [data_w for i in range(len(batch_filenames))]

            # next code will be processed only if for-loop has only one iteration
            if self._dictionary is not None:
                self._dictionary.gather(data_path=target_f)

    def _parse_batches(self, data_weight=None, batches=None):
        if self._process_in_memory:
            self._model.master.import_batches(batches)
            self._batches_list = [batch.id for batch in batches]
            return

        data_paths, data_weights, target_folders = self._populate_data(data_weight, True)
        for (data_p, data_w, target_f) in zip(data_paths, data_weights, target_folders):
            if batches is None:
                batch_filenames = glob.glob(os.path.join(data_p, '*.batch'))
                self._batches_list += [Batch(filename) for filename in batch_filenames]

                if len(self._batches_list) < 1:
                    raise RuntimeError('No batches were found')

                self._weights += [data_w for i in range(len(batch_filenames))]
            else:
                self._batches_list += [Batch(os.path.join(data_p, batch)) for batch in batches]
                self._weights += [data_w for i in range(len(batches))]

    def _parse_n_wd(self, data_weight=None, n_wd=None, vocab=None):
        def __reset_batch():
            batch = messages.Batch()
            batch.id = str(uuid.uuid4())
            return batch, {}

        try:
            from scipy.sparse.base import spmatrix
        except ImportError:
            spmatrix = tuple()

        os.mkdir(self._target_folder)
        global_vocab, global_n = {}, 0.0
        batch, batch_vocab = __reset_batch()
        try:
            n_wd_T = n_wd.T
        except AttributeError:
            raise TypeError("Expected a transposable matrix, got {}".format(type(n_wd)))
        for item_id, column in enumerate(n_wd_T):
            item = batch.item.add()
            item.id = item_id
            for key in global_vocab.keys():
                global_vocab[key][2] = False  # all tokens haven't appeared in this item yet

            if isinstance(column, np.matrix):
                enum = enumerate(np.squeeze(np.asarray(column), axis=0))
            elif isinstance(column, np.ndarray):
                enum = enumerate(column)
            elif isinstance(column, spmatrix):
                nnz = column.nonzero()[1]
                enum = zip(nnz, np.squeeze(column[0, nnz].toarray(), axis=0))
            else:
                raise TypeError("Unsupported column type: %s" % type(column))
            for token_id, value in enum:
                if value <= GLOB_EPS:
                    continue
                token = vocab[token_id]
                if token not in global_vocab:
                    global_vocab[token] = [0, 0, False]  # token_tf, token_df, appeared in this item

                global_vocab[token][0] += value
                global_vocab[token][1] += 0 if global_vocab[token][2] else 1
                global_n += value

                if token not in batch_vocab:
                    batch_vocab[token] = len(batch.token)
                    batch.token.append(token)

                item.token_id.append(batch_vocab[token])
                item.token_weight.append(float(value))

            if ((item_id + 1) % self._batch_size == 0 and item_id != 0) or ((item_id + 1) == n_wd.shape[1]):
                filename = os.path.join(self._target_folder, '{}.batch'.format(batch.id))
                with open(filename, 'wb') as fout:
                    fout.write(batch.SerializeToString())
                batch, batch_vocab = __reset_batch()

        batch_filenames = glob.glob(os.path.join(self._target_folder, '*.batch'))
        self._batches_list += [Batch(filename) for filename in batch_filenames]
        self._weights += [data_weight for i in range(len(batch_filenames))]

        dictionary_data = messages.DictionaryData()
        dictionary_data.name = uuid.uuid1().urn.replace(':', '')
        for key, value in iteritems(global_vocab):
            dictionary_data.token.append(key)
            dictionary_data.token_tf.append(int(value[0]))
            dictionary_data.token_df.append(int(value[1]))
            dictionary_data.token_value.append(float(value[0]) / global_n)

        self._dictionary.create(dictionary_data)

    @property
    def batches_ids(self):
        """
        :return: list of batches filenames, if process_in_memory == False,\
                 else - the list of in memory batches ids
        """
        if self._process_in_memory:
            return self._batches_list
        else:
            return [batch.filename for batch in self._batches_list]

    @property
    def batches_list(self):
        """
        : return: list of batches, if process_in_memory == False,\
                  else - the list of in memory batches ids
        """
        return self._batches_list

    @property
    def weights(self):
        """
        :return: list of batches weights
        """
        return self._weights

    @property
    def num_batches(self):
        """
        :return: the number of batches
        """
        return len(self._batches_list)

    @property
    def data_path(self):
        """
        :return: the disk path of batches
        """
        return self._data_path

    @property
    def batch_size(self):
        """
        :return: the user-defined size of the batches
        """
        return self._batch_size

    @property
    def dictionary(self):
        """
        :return: Dictionary object, if parameter gather_dictionary was True, else None
        """
        return self._dictionary

    @property
    def process_in_memory(self):
        """
        :return: if Vectorizer uses processing of batches in core memory
        """
        return self._process_in_memory

    def __repr__(self):
        return 'artm.BatchVectorizer(data_path="{0}", num_batches={1})'.format(
            self._data_path, self.num_batches)
