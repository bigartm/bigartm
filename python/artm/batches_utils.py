import os
import glob

from . import wrapper
from wrapper import constants as const
from wrapper import messages_pb2 as messages


__all__ = [
    'BatchVectorizer'
]


class Batch(object):
    def __init__(self, filename):
        self._filename = os.path.abspath(filename)

    def __repr__(self):
        return 'Batch({0})'.format(self._filename)

    @property
    def filename(self):
        return self._filename


class BatchVectorizer(object):
    def __init__(self, batches=None, collection_name=None, data_path='', data_format='batches',
                 target_folder='', batch_size=1000, batch_name_type='code', data_weight=1.0):
        """
        :param str collection_name: the name of text collection (required if data_format == 'bow_uci')
        :param str data_path: 1) if data_format == 'bow_uci' => folder containing\
                                 'docword.collection_name.txt' and vocab.collection_name.txt files;\
                              2) if data_format == 'vowpal_wabbit' => file in Vowpal Wabbit format;\
                              3) if data_format == 'plain_text' => file with text;\
                              4) if data_format == 'batches' => folder containing batches
        :param str data_format: the type of input data:\
                              1) 'bow_uci' --- Bag-Of-Words in UCI format;\
                              2) 'vowpal_wabbit' --- Vowpal Wabbit format;\
                              3) 'batches' --- the BigARTM data format
        :param int batch_size: number of documents to be stored in each batch
        :param str target_folder: full path to folder for future batches storing
        :param batches: list with non-full file names of batches (necessary parameters are\
                              batches + data_path + data_fromat=='batches' in this case)
        :type batches: list of str
        :param str batch_name_type: name batches in natural order ('code') or using random guids (guid)
        :param float data_weight: weight for a group of batches from data_path;\
                              it can be a list of floats, then data_path (and\
                              target_folder if not data_format == 'batches')\
                              should also be lists; one weight corresponds to\
                              one path from the data_path list;
        """
        self._batches_list = []
        self._weights = []
        self._data_path = data_path if data_format == 'batches' else target_folder
        self._batch_size = batch_size

        if isinstance(data_path, list):
            data_paths = data_path
            if len(data_path) != len(data_weight):
                raise IOError('Lists for data_path and data_weight should have the same length')
            data_weights = data_weight
            if data_format == 'batches':
                target_folders = ['' for p in data_paths]
            else:
                if len(data_path) != len(target_folder):
                    raise IOError('Lists for data_path and target_folder should have same length')
                target_folders = target_folder
        else:
            if isinstance(data_weight, list) or isinstance(target_folder, list):
                raise IOError('data_path should be also a list for multiple weights or folders')
            data_paths = [data_path]
            data_weights = [data_weight]
            target_folders = [target_folder]

        for (data_path, data_weight, target_folder) in zip(data_paths, data_weights,
                                                           target_folders):
            if data_format == 'batches':
                if batches is None:
                    batch_filenames = glob.glob(os.path.join(data_path, '*.batch'))
                    self._batches_list += [Batch(filename) for filename in batch_filenames]
                    if len(self._batches_list) < 1:
                        raise RuntimeError('No batches were found')
                    self._weights += [data_weight for i in xrange(len(batch_filenames))]
                else:
                    self._batches_list += [Batch(os.path.join(data_path, batch))
                                           for batch in batches]
                    self._weights += [data_weight for i in xrange(len(batches))]

            elif data_format == 'bow_uci' or data_format == 'vowpal_wabbit':
                parser_config = messages.CollectionParserConfig()
                parser_config.num_items_per_batch = batch_size

                parser_config.name_type = const.CollectionParserConfig_NameType_Code
                if batch_name_type == 'guid':
                    parser_config.name_type = const.CollectionParserConfig_NameType_Guid

                if data_format == 'bow_uci':
                    parser_config.docword_file_path = os.path.join(
                        data_path, 'docword.{0}.txt'.format(collection_name))
                    parser_config.vocab_file_path = os.path.join(
                        data_path, 'vocab.{0}.txt'.format(collection_name))
                    parser_config.format = const.CollectionParserConfig_CollectionFormat_BagOfWordsUci
                elif data_format == 'vowpal_wabbit':
                    parser_config.docword_file_path = data_path
                    parser_config.format = const.CollectionParserConfig_CollectionFormat_VowpalWabbit
                parser_config.target_folder = target_folder

                lib = wrapper.LibArtm()
                lib.ArtmParseCollection(parser_config)
                batch_filenames = glob.glob(os.path.join(target_folder, '*.batch'))
                self._batches_list += [Batch(filename) for filename in batch_filenames]
                self._weights += [data_weight for i in xrange(len(batch_filenames))]
            else:
                raise IOError('Unknown data format')

    @property
    def batches_list(self):
        """
        :return: list of batches names
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
