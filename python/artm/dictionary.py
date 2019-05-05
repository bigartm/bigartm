# Copyright 2017, Additive Regularization of Topic Models.

import uuid
import os
import glob
import codecs

from six.moves import range

from . import wrapper
from .wrapper import messages_pb2 as messages
from . import master_component

__all__ = [
    'Dictionary'
]


class Dictionary(object):
    def __init__(self, name=None, dictionary_path=None, data_path=None):
        """
        :param str name: name of the dictionary
        :param str dictionary_path: can be used for default call of load() method\
          in constructor
        :param str data_path: can be used for default call of gather() method\
          in constructor

        Note: all parameters are optional
        """
        self._name = name if name is not None else str(uuid.uuid4())
        self._lib = wrapper.LibArtm()
        self._master = master_component.MasterComponent(self._lib, num_processors=0)

        if dictionary_path is not None:
            self.load(dictionary_path=dictionary_path)
        if data_path is not None:
            self.gather(data_path=data_path)

    def __enter__(self):
        return self

    def dispose(self):
        if self._master is not None:
            self._reset()
            self._lib.ArtmDisposeMasterComponent(self._master.master_id)
            self._master = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.dispose()

    def __del__(self):
        self.dispose()

    @property
    def name(self):
        return self._name

    def _reset(self):
        self._lib.ArtmDisposeDictionary(self._master.master_id, self._name)

    def load(self, dictionary_path):
        """
        :Description: loads the BigARTM dictionary of the collection into the lib

        :param str dictionary_path: full filename of the dictionary
        """
        self._reset()
        self._master.import_dictionary(filename=dictionary_path, dictionary_name=self._name)

    def save(self, dictionary_path):
        """
        :Description: saves the BigARTM dictionary of the collection on the disk

        :param str dictionary_path: full file name for the dictionary
        """
        self._master.export_dictionary(filename=dictionary_path, dictionary_name=self._name)

    def save_text(self, dictionary_path, encoding='utf-8'):
        """
        :Description: saves the BigARTM dictionary of the collection on the disk\
                      in the human-readable text format

        :param str dictionary_path: full file name for the text dictionary file
        :param str encoding: an encoding of text in diciotnary
        """
        dictionary_data = self._master.get_dictionary(self._name)
        with codecs.open(dictionary_path, 'w', encoding) as fout:
            fout.write(u'name: {} num_items: {}\n'.format(dictionary_data.name,
                                                          dictionary_data.num_items_in_collection))
            fout.write(u'token, class_id, token_value, token_tf, token_df\n')

            for i in range(len(dictionary_data.token)):
                fout.write(u'{0}, {1}, {2}, {3}, {4}\n'.format(dictionary_data.token[i],
                                                               dictionary_data.class_id[i],
                                                               dictionary_data.token_value[i],
                                                               dictionary_data.token_tf[i],
                                                               dictionary_data.token_df[i]))

    def load_text(self, dictionary_path, encoding='utf-8'):
        """
        :Description: loads the BigARTM dictionary of the collection from the disk\
                      in the human-readable text format

        :param str dictionary_path: full file name of the text dictionary file
        :param str encoding: an encoding of text in diciotnary
        """
        self._reset()
        dictionary_data = messages.DictionaryData()
        with codecs.open(dictionary_path, 'r', encoding) as fin:
            first_str = fin.readline()[: -1].split(' ')
            dictionary_data.name = first_str[1]
            dictionary_data.num_items_in_collection = int(first_str[3])
            fin.readline()  # skip comment line

            for line in fin:
                line_list = line.split(' ')
                dictionary_data.token.append(line_list[0][0: -1])
                dictionary_data.class_id.append(line_list[1][0: -1])
                dictionary_data.token_value.append(float(line_list[2][0: -1]))
                dictionary_data.token_tf.append(float(line_list[3][0: -1]))
                dictionary_data.token_df.append(float(line_list[4][0: -1]))

        self._master.create_dictionary(dictionary_data=dictionary_data, dictionary_name=self._name)

    def create(self, dictionary_data):
        """
        :Description: creates dictionary using DictionaryData object

        :param dictionary_data: configuration of dictionary
        :type dictionary_data: DictionaryData instance
        """
        self._reset()
        self._master.create_dictionary(dictionary_data=dictionary_data, dictionary_name=self._name)

    def gather(self, data_path, cooc_file_path=None, vocab_file_path=None, symmetric_cooc_values=False):
        """
        :Description: creates the BigARTM dictionary of the collection,\
                      represented as batches and load it in the lib

        :param str data_path: full path to batches folder
        :param str cooc_file_path: full path to the file with cooc info. Cooc info is a file with three\
                                   columns, first two a the zero-based indices of tokens in vocab file,\
                                   and third one is a value of their co-occurrence in collection (or another)\
                                   pairwise statistic.
        :param str vocab_file_path: full path to the file with vocabulary.\
                      If given, the dictionary token will have the same order, as in\
                      this file, otherwise the order will be random.\
                      If given, the tokens from batches, that are not presented in vocab, will be skipped.
        :param bool symmetric_cooc_values: if the cooc matrix should considered\
                      to be symmetric or not
        """
        self._reset()
        self._master.gather_dictionary(dictionary_target_name=self._name,
                                       data_path=data_path,
                                       cooc_file_path=cooc_file_path,
                                       vocab_file_path=vocab_file_path,
                                       symmetric_cooc_values=symmetric_cooc_values)

    def filter(self, class_id=None, min_df=None, max_df=None, min_df_rate=None, max_df_rate=None,
               min_tf=None, max_tf=None, max_dictionary_size=None, recalculate_value=False, inplace=True):
        """
        :Description: filters the BigARTM dictionary of the collection, which\
                      was already loaded into the lib

        :param str class_id: class_id to filter
        :param float min_df: min df value to pass the filter
        :param float max_df: max df value to pass the filter
        :param float min_df_rate: min df rate to pass the filter
        :param float max_df_rate: max df rate to pass the filter
        :param float min_tf: min tf value to pass the filter
        :param float max_tf: max tf value to pass the filter
        :param float max_dictionary_size: give an easy option to limit dictionary size;
                                          rare tokens will be excluded until dictionary reaches given size.
        :param bool recalculate_value: recalculate or not value field in dictionary after filtration\
                                       according to new sun of tf values
        :param bool inplace: if True, fill in place, otherwise return a new dictionary

        :Note: the current dictionary will be replaced with filtered
        """
        target = self if inplace else Dictionary()
        self._master.filter_dictionary(dictionary_target_name=target._name,
                                       dictionary_name=self._name,
                                       class_id=class_id,
                                       min_df=min_df,
                                       max_df=max_df,
                                       min_df_rate=min_df_rate,
                                       max_df_rate=max_df_rate,
                                       min_tf=min_tf,
                                       max_tf=max_tf,
                                       max_dictionary_size=max_dictionary_size,
                                       recalculate_value=recalculate_value)
        return target

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        descr = next(x for x in self._master.get_info().dictionary if x.name == self.name)
        return 'artm.Dictionary(name={0}, num_entries={1})'.format(descr.name, descr.num_entries)
