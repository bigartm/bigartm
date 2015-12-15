import os
import glob

from . import wrapper
from wrapper import constants as const
from wrapper import messages_pb2 as messages

DICTIONARY_NAME = 'dictionary'


class Batch(object):
    def __init__(self, filename):
        self._filename = os.path.abspath(filename)

    def __repr__(self):
        return 'Batch({0})'.format(self._filename)

    @property
    def filename(self):
        return self._filename


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
      target_folder(str): full path to folder for future batches storing
      batches(list of str): list with non-full file names of batches (necessary parameters are
      batches + data_path + data_fromat=='batches' in this case)
    """
    def __init__(self, batches=None, collection_name=None, data_path='', data_format='batches',
                 target_folder='', batch_size=1000):
        self._batches_list = []
        if data_format == 'batches':
            if batches is None:
                batch_filenames = glob.glob(os.path.join(data_path, '*.batch'))
                self._batches_list = [Batch(filename) for filename in batch_filenames]
                if len(self._batches_list) < 1:
                    raise RuntimeError('No batches were found')
            else:
                self._batches_list = [Batch(os.path.join(data_path, batch)) for batch in batches]

        elif data_format == 'bow_uci' or data_format == 'vowpal_wabbit':
            parser_config = messages.CollectionParserConfig()
            parser_config.num_items_per_batch = batch_size
            if data_format == 'bow_uci':
                parser_config.docword_file_path = os.path.join(
                    data_path, 'docword.{0}.txt'.format(collection_name))
                parser_config.vocab_file_path = os.path.join(
                    data_path, 'vocab.{0}.txt'.format(collection_name))
                parser_config.format = const.CollectionParserConfig_Format_BagOfWordsUci
            elif data_format == 'vowpal_wabbit':
                parser_config.docword_file_path = data_path
                parser_config.format = const.CollectionParserConfig_Format_VowpalWabbit
            parser_config.target_folder = target_folder

            lib = wrapper.LibArtm()
            lib.ArtmParseCollection(parser_config)
            batch_filenames = glob.glob(os.path.join(target_folder, '*.batch'))
            self._batches_list = [Batch(filename) for filename in batch_filenames]

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
