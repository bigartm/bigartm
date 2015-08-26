import os

from . import wrapper
from wrapper import constants
from wrapper import messages_pb2 as messages
from . import master_component as mc

DICTIONARY_NAME = 'dictionary'


def _create_parser_config(data_path, collection_name, target_folder,
                          batch_size, data_format, dictionary_name=DICTIONARY_NAME):
    collection_parser_config = messages.CollectionParserConfig()
    collection_parser_config.num_items_per_batch = batch_size
    if data_format == 'bow_uci':
        collection_parser_config.docword_file_path = \
            os.path.join(data_path, 'docword.' + collection_name + '.txt')
        collection_parser_config.vocab_file_path = \
            os.path.join(data_path, 'vocab.' + collection_name + '.txt')
        collection_parser_config.format = constants.CollectionParserConfig_Format_BagOfWordsUci
    elif data_format == 'vowpal_wabbit':
        collection_parser_config.docword_file_path = data_path
        collection_parser_config.format = constants.CollectionParserConfig_Format_VowpalWabbit
    collection_parser_config.target_folder = target_folder
    collection_parser_config.dictionary_file_name = dictionary_name

    return collection_parser_config


def parse(collection_name=None, data_path='', data_format='bow_uci',
          batch_size=1000, dictionary_name=DICTIONARY_NAME):
    """parse() --- proceed the learning of topic model

    Args:
      collection_name (str): the name of text collection (required if
      data_format == 'bow_uci'), default=None
      data_path (str):
      1) if data_format == 'bow_uci' => folder containing
      'docword.collection_name.txt' and vocab.collection_name.txt files;
      2) if data_format == 'vowpal_wabbit' => file in Vowpal Wabbit format;
      3) if data_format == 'plain_text' => file with text;
      default=''
      data_format (str:) the type of input data;
      1) 'bow_uci' --- Bag-Of-Words in UCI format;
      2) 'vowpal_wabbit' --- Vowpal Wabbit format;
      3) 'plain_text' --- source text;
      default='bow_uci'
      batch_size (int): number of documents to be stored in each batch,
      default=1000
      dictionary_name (str): the name of BigARTM dictionary with information
      about collection, that will be gathered by the library parser;
      default=DICTIONARY_NAME
    """
    if collection_name is None and data_format == 'bow_uci':
        raise IOError('ArtmModel.parse(): No collection name was given')

    lib = wrapper.LibArtm()
    if data_format == 'bow_uci' or data_format == 'vowpal_wabbit':
        collection_parser_config = _create_parser_config(
            data_path,
            collection_name,
            collection_name,
            batch_size,
            data_format,
            dictionary_name)
        lib.ArtmParseCollection(collection_parser_config)

    elif data_format == 'plain_text':
        raise NotImplementedError()
    else:
        raise IOError('parse(): Unknown data format')
