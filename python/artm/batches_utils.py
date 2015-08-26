import os
import glob

from . import wrapper
from wrapper import constants
from wrapper import messages_pb2 as messages

DICTIONARY_NAME = 'dictionary'


class Batch(object):
    def __init__(self, file_name):
        self._file_name = file_name

    def __str__(self):
        return self._file_name


class BatchVectorizer(object):
    """BatchVectorizer() --- class, represents the general type of ARTM input data.
    Args:
      collection_name (str): the name of text collection (required if
      data_format == 'bow_uci'), default=None
      data_path (str):
      1) if data_format == 'bow_uci' => folder containing
      'docword.collection_name.txt' and vocab.collection_name.txt files;
      2) if data_format == 'vowpal_wabbit' => file in Vowpal Wabbit format;
      3) if data_format == 'plain_text' => file with text;
      4) if data_format == 'batches' => folder containing batches
      default=''
      data_format (str:) the type of input data;
      1) 'bow_uci' --- Bag-Of-Words in UCI format;
      2) 'vowpal_wabbit' --- Vowpal Wabbit format;
      3) 'plain_text' --- source text;
      4) 'batches' --- the BigARTM data format
      default='batches'
      batch_size (int): number of documents to be stored in each batch,
      default=1000
      dictionary_name (str): the name of BigARTM dictionary with information
      about collection, that will be gathered by the library parser;
      default=DICTIONARY_NAME
      target_folder(str): full path to folder for future batches storing
      batches(list of str): list with non-full file names of batches (necessary parameters are
      batches + data_path + data_fromat=='batches' in this case)
    """
    def __init__(self, batches=None, collection_name=None, data_path='', data_format='batches',
                 target_folder='', batch_size=1000, dictionary_name=DICTIONARY_NAME):
        self._batches_list = []
        if data_format == 'batches':
            if batches is None:
                batch_str = glob.glob(os.path.join(data_path, '*.batch'))
                self._batches_list = [Batch(batch) for batch in batch_str]
                if len(self._batches_list) < 1:
                    raise RuntimeError('No batches were found')
            else:
                self._batches_list = [Batch(os.path.join(data_path, batch)) for batch in batches]

        elif data_format == 'bow_uci' or data_format == 'vowpal_wabbit':
            parser_config = messages.CollectionParserConfig()
            parser_config.num_items_per_batch = batch_size
            if data_format == 'bow_uci':
                parser_config.docword_file_path =\
                    os.path.join(data_path, 'docword.' + collection_name + '.txt')
                parser_config.vocab_file_path =\
                    os.path.join(data_path, 'vocab.' + collection_name + '.txt')
                parser_config.format = constants.CollectionParserConfig_Format_BagOfWordsUci
            elif data_format == 'vowpal_wabbit':
                parser_config.docword_file_path = data_path
                parser_config.format = constants.CollectionParserConfig_Format_VowpalWabbit
            parser_config.target_folder = target_folder
            parser_config.dictionary_file_name = dictionary_name

            lib = wrapper.LibArtm()
            lib.ArtmParseCollection(parser_config)
            batch_str = glob.glob(os.path.join(target_folder, '*.batch'))
            self._batches_list = [Batch(batch) for batch in batch_str]

        elif data_format == 'plain_text':
            raise NotImplementedError()
        else:
            raise IOError('Unknown data format')

        self._data_path = data_path if data_format == 'batches' else target_folder
        self._batch_size = batch_size

    @property
    def batches_list(self):
        return self._batches_list

    @property
    def data_path(self):
        return self._data_path

    @property
    def batch_size(self):
        return self._batch_size
