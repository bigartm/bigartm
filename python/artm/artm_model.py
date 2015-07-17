"""This module contains ArtmModel class and helper classes and provides high-level
Python API for BigARTM Library.

Written for Python 2.7
Code satisfies the pep8 Python code style guide.

Each change of this file should be tested with pep8 Python style guide checker
(https://pypi.python.org/pypi/pep8) using command
> pep8 --first --max-line-length=99 artm_model.py
"""

import os
import csv
import uuid
import glob
import shutil
import random
import collections

from pandas import DataFrame

import artm.messages_pb2 as messages_pb2
import artm.library as library


###################################################################################################
THETA_REGULARIZER_TYPE = 0
PHI_REGULARIZER_TYPE = 1
GLOB_EPS = 1e-37


def reconfigure_score_in_master(master, score_config, name):
    """reconfigure_score_in_master --- helpful internal method"""
    master_config = messages_pb2.MasterComponentConfig()
    master_config.CopyFrom(master.config())
    for i in xrange(len(master_config.score_config)):
        if master_config.score_config[i].name == name:
            master_config.score_config[i].config = score_config.SerializeToString()
            break
    master.Reconfigure(master_config)


def create_parser_config(data_path, collection_name, target_folder,
                         batch_size, data_format, dictionary_name='dictionary'):
    """create_parser_config --- helpful internal method"""
    collection_parser_config = messages_pb2.CollectionParserConfig()
    collection_parser_config.num_items_per_batch = batch_size
    if data_format == 'bow_uci':
        collection_parser_config.docword_file_path = \
            os.path.join(data_path, 'docword.' + collection_name + '.txt')
        collection_parser_config.vocab_file_path = \
            os.path.join(data_path, 'vocab.' + collection_name + '.txt')
        collection_parser_config.format = library.CollectionParserConfig_Format_BagOfWordsUci
    elif data_format == 'vowpal_wabbit':
        collection_parser_config.docword_file_path = data_path
        collection_parser_config.format = library.CollectionParserConfig_Format_VowpalWabbit
    collection_parser_config.target_folder = target_folder
    collection_parser_config.dictionary_file_name = dictionary_name

    return collection_parser_config


###################################################################################################
def parse(collection_name=None, data_path='', data_format='bow_uci',
          batch_size=1000, dictionary_name='dictionary'):
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
      default='dictionary'
    """
    if collection_name is None and data_format == 'bow_uci':
        raise IOError('ArtmModel.parse(): No collection name was given')

    if data_format == 'bow_uci' or data_format == 'vowpal_wabbit':
        collection_parser_config = create_parser_config(data_path,
                                                        collection_name,
                                                        collection_name,
                                                        batch_size,
                                                        data_format,
                                                        dictionary_name)
        library.Library().ParseCollection(collection_parser_config)

    elif data_format == 'plain_text':
        raise NotImplementedError()
    else:
        raise IOError('ArtmModel.parse(): Unknown data format')


###################################################################################################
class Regularizers(object):
    """Regularizers represents a storage of regularizers in ArtmModel (private class)

    Args:
      master (reference): reference to MasterComponent object, no default
    """

    def __init__(self, master):
        self._data = {}
        self._master = master

    def add(self, config):
        """Regularizers.add() --- add regularizer into ArtmModel

        Args:
          config (reference): reference to an object of ***Regularizer class, no default
        """
        if config.name in self._data:
            raise ValueError('Regularizer with name ' + str(config.name) + ' is already exist')
        else:
            regularizer = self._master.CreateRegularizer(config.name, config.type, config.config)
            config._regularizer = regularizer
            self._data[config.name] = config

    def __getitem__(self, name):
        """Regularizers.__getitem__() --- get regularizer with given name

        Args:
          name (str): name of the regularizer, no default
        """
        if name in self._data:
            return self._data[name]
        else:
            raise KeyError('No regularizer with name ' + name)

    @property
    def data(self):
        return self._data


###################################################################################################
class Scores(object):
    """Scores represents a storage of scores in ArtmModel (private class)

    Args:
      master (reference): reference to MasterComponent object, no default
    """

    def __init__(self, master, model):
        self._data = {}
        self._master = master
        self._model = model

    def add(self, config):
        """Scores.add() --- add score into ArtmModel.

        Args:
          config (reference): an object of ***Scores class, no default
        """
        if config.name in self._data:
            raise ValueError('Score with name ' + str(config.name) + ' is already exist')
        else:
            score = self._master.CreateScore(config.name, config.type, config.config)
            config._model = self._model
            config._score = score
            config._master = self._master
            self._data[config.name] = config

    def __getitem__(self, name):
        """Scores.__getitem__() --- get score with given name

        Args:
          name (str): name of the regularizer, no default
        """
        if name in self._data:
            return self._data[name]
        else:
            raise KeyError('No score with name ' + name)

    @property
    def data(self):
        return self._data


###################################################################################################
# SECTION OF REGULARIZER CLASSES
###################################################################################################
class SmoothSparsePhiRegularizer(object):
    """SmoothSparsePhiRegularizer is a regularizer in ArtmModel (public class)

    Args:
      name (str): the identifier of regularizer, will be auto-generated if not specified
      tau (double): the coefficient of regularization for this regularizer, default=1.0
      class_ids (list of str): list of class_ids to regularize, will regularize all
      classes if not specified
      topic_names (list of str): list of names of topics to regularize, will regularize
      all topics if not specified
      dictionary_name (str): BigARTM collection dictionary, won't use dictionary if not
      specified
    """

    def __init__(self, name=None, tau=1.0, class_ids=None,
                 topic_names=None, dictionary_name=None):
        config = messages_pb2.SmoothSparsePhiConfig()
        self._class_ids = []
        self._topic_names = []
        self._dictionary_name = ''

        if name is None:
            name = 'SmoothSparsePhiRegularizer:' + uuid.uuid1().urn
        if class_ids is not None:
            config.ClearField('class_id')
            for class_id in class_ids:
                config.class_id.append(class_id)
                self._class_ids.append(class_id)
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
                self._topic_names.append(topic_name)
        if dictionary_name is not None:
            config.dictionary_name = dictionary_name
            self._dictionary_name = dictionary_name

        self._name = name
        self.tau = tau
        self._type = PHI_REGULARIZER_TYPE
        self._config = config
        self._type = library.RegularizerConfig_Type_SmoothSparsePhi
        self._regularizer = None  # Reserve place for the regularizer

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def config(self):
        return self._config

    @property
    def class_ids(self):
        return self._class_ids

    @property
    def topic_names(self):
        return self._topic_names

    @property
    def dictionary_name(self):
        return self._dictionary_name

    @property
    def regularizer(self):
        return self._regularizer

    @class_ids.setter
    def class_ids(self, class_ids):
        self._class_ids = class_ids
        config = messages_pb2.SmoothSparsePhiConfig()
        config.CopyFrom(self._config)
        config.ClearField('class_id')
        for class_id in class_ids:
            config.class_id.append(class_id)
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @topic_names.setter
    def topic_names(self, topic_names):
        self._topic_names = topic_names
        config = messages_pb2.SmoothSparsePhiConfig()
        config.CopyFrom(self._config)
        config.ClearField('topic_name')
        for topic_name in topic_names:
            config.topic_name.append(topic_name)
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @dictionary_name.setter
    def dictionary_name(self, dictionary_name):
        self._dictionary_name = dictionary_name
        config = messages_pb2.SmoothSparsePhiConfig()
        config.CopyFrom(self._config)
        config.dictionary_name = dictionary_name
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)


###################################################################################################
class SmoothSparseThetaRegularizer(object):
    """SmoothSparseThetaRegularizer is a regularizer in ArtmModel (public class)

    Args:
      name (str): the identifier of regularizer, will be auto-generated if not specified
      tau (double): the coefficient of regularization for this regularizer, default=1.0
      topic_names (list of str): list of names of topics to regularize, will regularize
      all topics if not specified
      alpha_iter (list of double, default=None): list of additional coefficients of
      regularization on each iteration over document. Should have length equal to
      model.num_document_passes
    """

    def __init__(self, name=None, tau=1.0, topic_names=None, alpha_iter=None):
        config = messages_pb2.SmoothSparseThetaConfig()
        self._topic_names = []
        self._alpha_iter = []

        if name is None:
            name = 'SmoothSparseThetaRegularizer:' + uuid.uuid1().urn
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
                self._topic_names.append(topic_name)
        if alpha_iter is not None:
            config.ClearField('alpha_iter')
            for alpha in alpha_iter:
                config.alpha_iter.append(alpha)
                self._alpha_iter.append(alpha)

        self._name = name
        self.tau = tau
        self._type = THETA_REGULARIZER_TYPE
        self._config = config
        self._type = library.RegularizerConfig_Type_SmoothSparseTheta
        self._regularizer = None  # Reserve place for the regularizer

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def config(self):
        return self._config

    @property
    def topic_names(self):
        return self._topic_names

    @property
    def alpha_iter(self):
        return self._alpha_iter

    @property
    def regularizer(self):
        return self._regularizer

    @topic_names.setter
    def topic_names(self, topic_names):
        self._topic_names = topic_names
        config = messages_pb2.SmoothSparseThetaConfig()
        config.CopyFrom(self._config)
        config.ClearField('topic_name')
        for topic_name in topic_names:
            config.topic_name.append(topic_name)
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @alpha_iter.setter
    def alpha_iter(self, alpha_iter):
        self._alpha_iter = alpha_iter
        config = messages_pb2.SmoothSparseThetaConfig()
        config.CopyFrom(self._config)
        config.ClearField('alpha_iter')
        for alpha in alpha_iter:
            config.alpha_iter.append(alpha)
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)


###################################################################################################
class DecorrelatorPhiRegularizer(object):
    """DecorrelatorPhiRegularizer is a regularizer in ArtmModel (public class)

    Args:
      name (str): the identifier of regularizer, will be auto-generated if not specified
      tau (double): the coefficient of regularization for this regularizer, default=1.0
      class_ids (list of str): list of class_ids to regularize, will regularize all
      classes if not specified
      topic_names (list of str): list of names of topics to regularize, will regularize
      all topics if not specified
    """

    def __init__(self, name=None, tau=1.0, class_ids=None, topic_names=None):
        config = messages_pb2.DecorrelatorPhiConfig()
        self._class_ids = []
        self._topic_names = []

        if name is None:
            name = 'DecorrelatorPhiRegularizer:' + uuid.uuid1().urn
        if class_ids is not None:
            config.ClearField('class_id')
            for class_id in class_ids:
                config.class_id.append(class_id)
                self._class_ids.append(class_id)
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
                self._topic_names.append(topic_name)

        self._name = name
        self.tau = tau
        self._type = PHI_REGULARIZER_TYPE
        self._config = config
        self._type = library.RegularizerConfig_Type_DecorrelatorPhi
        self._regularizer = None  # Reserve place for the regularizer

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def config(self):
        return self._config

    @property
    def class_ids(self):
        return self._class_ids

    @property
    def topic_names(self):
        return self._topic_names

    @property
    def regularizer(self):
        return self._regularizer

    @class_ids.setter
    def class_ids(self, class_ids):
        self._class_ids = class_ids
        config = messages_pb2.DecorrelatorPhiConfig()
        config.CopyFrom(self._config)
        config.ClearField('class_id')
        for class_id in class_ids:
            config.class_id.append(class_id)
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @topic_names.setter
    def topic_names(self, topic_names):
        self._topic_names = topic_names
        config = messages_pb2.DecorrelatorPhiConfig()
        config.CopyFrom(self._config)
        config.ClearField('topic_name')
        for topic_name in topic_names:
            config.topic_name.append(topic_name)
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)


###################################################################################################
class LabelRegularizationPhiRegularizer(object):
    """LabelRegularizationPhiRegularizer is a regularizer in ArtmModel (public class)

    Args:
      name (str): the identifier of regularizer, will be auto-generated if not specified
      tau (double): the coefficient of regularization for this regularizer, default=1.0
      class_ids (list of str): list of class_ids to regularize, will regularize all
      classes if not specified
      topic_names (list of str): list of names of topics to regularize, will regularize
      all topics if not specified
      dictionary_name (str): BigARTM collection dictionary, won't use dictionary if not
      specified
    """

    def __init__(self, name=None, tau=1.0, class_ids=None,
                 topic_names=None, dictionary_name=None):
        config = messages_pb2.LabelRegularizationPhiConfig()
        self._class_ids = []
        self._topic_names = []
        self._dictionary_name = ''

        if name is None:
            name = 'LabelRegularizationPhiRegularizer:' + uuid.uuid1().urn
        if class_ids is not None:
            config.ClearField('class_id')
            for class_id in class_ids:
                config.class_id.append(class_id)
                self._class_ids.append(class_id)
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
                self._topic_names.append(topic_name)
        if dictionary_name is not None:
            config.dictionary_name = dictionary_name
            self._dictionary_name = dictionary_name

        self._name = name
        self.tau = tau
        self._type = PHI_REGULARIZER_TYPE
        self._config = config
        self._type = library.RegularizerConfig_Type_LabelRegularizationPhi
        self._regularizer = None  # Reserve place for the regularizer

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def config(self):
        return self._config

    @property
    def class_ids(self):
        return self._class_ids

    @property
    def topic_names(self):
        return self._topic_names

    @property
    def dictionary_name(self):
        return self._dictionary_name

    @property
    def regularizer(self):
        return self._regularizer

    @class_ids.setter
    def class_ids(self, class_ids):
        self._class_ids = class_ids
        config = messages_pb2.LabelRegularizationPhiConfig()
        config.CopyFrom(self._config)
        config.ClearField('class_id')
        for class_id in class_ids:
            config.class_id.append(class_id)
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @topic_names.setter
    def topic_names(self, topic_names):
        self._topic_names = topic_names
        config = messages_pb2.LabelRegularizationPhiConfig()
        config.CopyFrom(self._config)
        config.ClearField('topic_name')
        for topic_name in topic_names:
            config.topic_name.append(topic_name)
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @dictionary_name.setter
    def dictionary_name(self, dictionary_name):
        self._dictionary_name = dictionary_name
        config = messages_pb2.LabelRegularizationPhiConfig()
        config.CopyFrom(self._config)
        config.dictionary_name = dictionary_name
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)


###################################################################################################
class SpecifiedSparsePhiRegularizer(object):
    """SpecifiedSparsePhiRegularizer is a regularizer in ArtmModel (public class)

    Args:
      name (str): the identifier of regularizer, will be auto-generated if not specified
      tau (double): the coefficient of regularization for this regularizer, default=1.0
      class_id (str): class_id to regularize, default=None
      topic_names (list of str): list of names of topics to regularize, will regularize
      all topics if not specified
      topic_names (list of str, default=None): list of names of topics to regularize
      num_max_elements (int): number of elements to save in row/column, default=None
      probability_threshold (double): if m elements in row/column
      summarize into value >= probability_threshold, m < n => only these elements would
      be saved. Value should be in (0, 1), default=None
      sparse_by_columns (bool) --- find max elements in column or in row, default=True
    """

    def __init__(self, name=None, tau=1.0, class_id=None, topic_names=None,
                 num_max_elements=None, probability_threshold=None, sparse_by_columns=True):
        config = messages_pb2.SpecifiedSparsePhiConfig()
        self._class_id = '@default_class'
        self._topic_names = []
        self._num_max_elements = 20
        self._probability_threshold = 0.99
        self._sparse_by_columns = True

        if name is None:
            name = 'SpecifiedSparsePhiRegularizer:' + uuid.uuid1().urn
        if class_id is not None:
            config.class_id = class_id
            self._class_id = class_id
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
                self._topic_names.append(topic_name)
        if num_max_elements is not None:
            config.max_elements_count = num_max_elements
            self._num_max_elements = num_max_elements
        if probability_threshold is not None:
            config.probability_threshold = probability_threshold
            self._probability_threshold = probability_threshold
        if sparse_by_columns is not None:
            if sparse_by_columns is True:
                config.mode = library.SpecifiedSparsePhiConfig_Mode_SparseTopics
                self._sparse_by_columns = True
            else:
                config.mode = library.SpecifiedSparsePhiConfig_Mode_SparseTokens
                self._sparse_by_columns = False

        self._name = name
        self.tau = tau
        self._type = PHI_REGULARIZER_TYPE
        self._config = config
        self._type = library.RegularizerConfig_Type_SpecifiedSparsePhi
        self._regularizer = None  # Reserve place for the regularizer

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def config(self):
        return self._config

    @property
    def class_id(self):
        return self._class_id

    @property
    def topic_names(self):
        return self._topic_names

    @property
    def num_max_elements(self):
        return self._num_max_elements

    @property
    def probability_threshold(self):
        return self._probability_threshold

    @property
    def sparse_by_columns(self):
        return self._sparse_by_columns

    @property
    def regularizer(self):
        return self._regularizer

    @class_id.setter
    def class_id(self, class_id):
        self._class_id = class_id
        config = messages_pb2.SpecifiedSparsePhiConfig()
        config.CopyFrom(self._config)
        config.class_id = class_id
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @topic_names.setter
    def topic_names(self, topic_names):
        self._topic_names = topic_names
        config = messages_pb2.SpecifiedSparsePhiConfig()
        config.CopyFrom(self._config)
        config.ClearField('topic_name')
        for topic_name in topic_names:
            config.topic_name.append(topic_name)
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @num_max_elements.setter
    def num_max_elements(self, num_max_elements):
        self._num_max_elements = num_max_elements
        config = messages_pb2.SpecifiedSparseRegularizationPhiConfig()
        config.CopyFrom(self._config)
        config.max_elements_count = num_max_elements
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @probability_threshold.setter
    def probability_threshold(self, probability_threshold):
        self._probability_threshold = probability_threshold
        config = messages_pb2.SpecifiedSparseRegularizationPhiConfig()
        config.CopyFrom(self._config)
        config.probability_threshold = probability_threshold
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @sparse_by_columns.setter
    def sparse_by_columns(self, sparse_by_columns):
        self._sparse_by_columns = sparse_by_columns
        config = messages_pb2.SpecifiedSparseRegularizationPhiConfig()
        config.CopyFrom(self._config)
        if sparse_by_columns is True:
            config.mode = library.SpecifiedSparsePhiConfig_Mode_SparseTopics
        else:
            config.mode = library.SpecifiedSparsePhiConfig_Mode_SparseTokens
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)


###################################################################################################
class ImproveCoherencePhiRegularizer(object):
    """ImproveCoherencePhiRegularizer is a regularizer in ArtmModel (public class)

    Args:
      name (str): the identifier of regularizer, will be auto-generated if not specified
      tau (double): the coefficient of regularization for this regularizer, default=1.0
      class_ids (list of str): list of class_ids to regularize, will regularize all
      classes if not specified
      topic_names (list of str): list of names of topics to regularize, will regularize
      all topics if not specified
      dictionary_name (str): BigARTM collection dictionary, won't use dictionary if not
      specified
    """

    def __init__(self, name=None, tau=1.0, class_ids=None,
                 topic_names=None, dictionary_name=None):
        config = messages_pb2.ImproveCoherencePhiConfig()
        self._class_ids = []
        self._topic_names = []
        self._dictionary_name = ''

        if name is None:
            name = 'ImproveCoherencePhiRegularizer:' + uuid.uuid1().urn
        if class_ids is not None:
            config.ClearField('class_id')
            for class_id in class_ids:
                config.class_id.append(class_id)
                self._class_ids.append(class_id)
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
                self._topic_names.append(topic_name)
        if dictionary_name is not None:
            config.dictionary_name = dictionary_name
            self._dictionary_name = dictionary_name

        self._name = name
        self.tau = tau
        self._type = PHI_REGULARIZER_TYPE
        self._config = config
        self._type = library.RegularizerConfig_Type_ImproveCoherencePhi
        self._regularizer = None  # Reserve place for the regularizer

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def config(self):
        return self._config

    @property
    def class_ids(self):
        return self._class_ids

    @property
    def topic_names(self):
        return self._topic_names

    @property
    def dictionary_name(self):
        return self._dictionary_name

    @property
    def regularizer(self):
        return self._regularizer

    @class_ids.setter
    def class_ids(self, class_ids):
        self._class_ids = class_ids
        config = messages_pb2.ImproveCoherencePhiConfig()
        config.CopyFrom(self._config)
        config.ClearField('class_id')
        for class_id in class_ids:
            config.class_id.append(class_id)
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @topic_names.setter
    def topic_names(self, topic_names):
        self._topic_names = topic_names
        config = messages_pb2.ImproveCoherencePhiConfig()
        config.CopyFrom(self._config)
        config.ClearField('topic_name')
        for topic_name in topic_names:
            config.topic_name.append(topic_name)
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @dictionary_name.setter
    def dictionary_name(self, dictionary_name):
        self._dictionary_name = dictionary_name
        config = messages_pb2.ImproveCoherencePhiConfig()
        config.CopyFrom(self._config)
        config.dictionary_name = dictionary_name
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)


###################################################################################################
# SECTION OF SCORE CLASSES
###################################################################################################
class SparsityPhiScore(object):
    """SparsityPhiScore is a score in ArtmModel (public class)

    Args:
      name (str): the identifier of score, will be auto-generated if not specified
      class_id (str): class_id to score, default=None
      topic_names (list of str): list of names of topics to regularize, will
      score all topics if not specified
      eps (double): the tolerance const, everything < eps
      considered to be zero, default=1e-37
    """

    def __init__(self, name=None, class_id=None, topic_names=None, eps=None):
        config = messages_pb2.SparsityPhiScoreConfig()
        self._class_id = '@default_class'
        self._topic_names = []
        self._eps = GLOB_EPS

        if name is None:
            name = 'SparsityPhiScore:' + uuid.uuid1().urn
        if class_id is not None:
            config.class_id = class_id
            self._class_id = class_id
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
                self._topic_names.append(topic_name)
        if eps is not None:
            config.eps = eps
            self._eps = eps

        self._name = name
        self._config = config
        self._type = library.ScoreConfig_Type_SparsityPhi
        self._model = None  # Reserve place for the model
        self._master = None  # Reserve place for the master (to reconfigure Scores)
        self._score = None  # Reserve place for the score

    @property
    def name(self):
        return self._name

    @property
    def config(self):
        return self._config

    @property
    def type(self):
        return self._type

    @property
    def class_id(self):
        return self._class_id

    @property
    def topic_names(self):
        return self._topic_names

    @property
    def eps(self):
        return self._eps

    @property
    def model(self):
        return self._model

    @property
    def score(self):
        return self._score

    @property
    def master(self):
        return self._master

    @class_id.setter
    def class_id(self, class_id):
        self._class_id = class_id
        score_config = messages_pb2.SparsityPhiScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.class_id = class_id
        reconfigure_score_in_master(self._master, score_config, self._name)

    @topic_names.setter
    def topic_names(self, topic_names):
        self._topic_names = topic_names
        score_config = messages_pb2.SparsityPhiScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.ClearField('topic_names')
        for topic_name in topic_names:
            score_config.topic_name.append(topic_name)
        reconfigure_score_in_master(self._master, score_config, self._name)

    @eps.setter
    def eps(self, eps):
        self._eps = eps
        score_config = messages_pb2.SparsityPhiScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.eps = eps
        reconfigure_score_in_master(self._master, score_config, self._name)


###################################################################################################
class SparsityThetaScore(object):
    """SparsityThetaScore is a score in ArtmModel (public class)

    Args:
      name (str): the identifier of score, will be auto-generated if not specified
      topic_names (list of str): list of names of topics to regularize, will
      score all topics if not specified
      eps (double): the tolerance const, everything < eps
      considered to be zero, default=1e-37
    """

    def __init__(self, name=None, topic_names=None, eps=None):
        config = messages_pb2.SparsityThetaScoreConfig()
        self._topic_names = []
        self._eps = GLOB_EPS

        if name is None:
            name = 'SparsityThetaScore:' + uuid.uuid1().urn
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
                self._topic_names.append(topic_name)
        if eps is not None:
            config.eps = eps
            self._eps = eps

        self._name = name
        self._config = config
        self._type = library.ScoreConfig_Type_SparsityTheta
        self._model = None  # Reserve place for the model
        self._master = None  # Reserve place for the master (to reconfigure Scores)
        self._score = None  # Reserve place for the score

    @property
    def name(self):
        return self._name

    @property
    def config(self):
        return self._config

    @property
    def type(self):
        return self._type

    @property
    def topic_names(self):
        return self._topic_names

    @property
    def eps(self):
        return self._eps

    @property
    def model(self):
        return self._model

    @property
    def score(self):
        return self._score

    @property
    def master(self):
        return self._master

    @topic_names.setter
    def topic_names(self, topic_names):
        self._topic_names = topic_names
        score_config = messages_pb2.SparsityThetaScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.ClearField('topic_names')
        for topic_name in topic_names:
            score_config.topic_name.append(topic_name)
        reconfigure_score_in_master(self._master, score_config, self._name)

    @eps.setter
    def eps(self, eps):
        self._eps = eps
        score_config = messages_pb2.SparsityThetaScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.eps = eps
        reconfigure_score_in_master(self._master, score_config, self._name)


###################################################################################################
class PerplexityScore(object):
    """PerplexityScore is a score in ArtmModel (public class)

    Args:
      name (str): the identifier of score, will be auto-generated if not specified
      class_id (str): class_id to score, default=None
      topic_names (list of str): list of names of topics to regularize, will
      score all topics if not specified
      eps (double, default=None): the tolerance const, everything < eps
      considered to be zero
      dictionary_name (str): BigARTM collection dictionary, won't use dictionary
      if not specified
      use_unigram_document_model (bool): use uni-gram
      document/collection model if token's counter == 0, default=True
    """

    def __init__(self, name=None, class_id=None, topic_names=None, eps=None,
                 dictionary_name=None, use_unigram_document_model=None):
        config = messages_pb2.PerplexityScoreConfig()
        self._class_id = '@default_class'
        self._topic_names = []
        self._eps = GLOB_EPS
        self._dictionary_name = ''
        self._use_unigram_document_model = True

        if name is None:
            name = 'PerplexityScore:' + uuid.uuid1().urn
        if class_id is not None:
            config.class_id = class_id
            self._class_id = class_id
        if topic_names is not None:
            config.ClearField('theta_sparsity_topic_name')
            for topic_name in topic_names:
                config.theta_sparsity_topic_name.append(topic_name)
                self._topic_names.append(topic_name)
        if eps is not None:
            config.theta_sparsity_eps = eps
            self._eps = eps
        if dictionary_name is not None:
            self._dictionary_name = dictionary_name
            config.dictionary_name = dictionary_name
        if use_unigram_document_model is not None:
            self._use_unigram_document_model = use_unigram_document_model
            if use_unigram_document_model is True:
                config.model_type = library.PerplexityScoreConfig_Type_UnigramDocumentModel
            else:
                config.model_type = library.PerplexityScoreConfig_Type_UnigramCollectionModel

        self._name = name
        self._config = config
        self._type = library.ScoreConfig_Type_Perplexity
        self._model = None  # Reserve place for the model
        self._master = None  # Reserve place for the master (to reconfigure Scores)
        self._score = None  # Reserve place for the score

    @property
    def name(self):
        return self._name

    @property
    def config(self):
        return self._config

    @property
    def type(self):
        return self._type

    @property
    def class_id(self):
        return self._class_id

    @property
    def topic_names(self):
        return self._topic_names

    @property
    def eps(self):
        return self._eps

    @property
    def dictionary_name(self):
        return self._dictionary_name

    @property
    def use_unigram_document_model(self):
        return self._use_unigram_document_model

    @property
    def model(self):
        return self._model

    @property
    def score(self):
        return self._score

    @property
    def master(self):
        return self._master

    @class_id.setter
    def class_id(self, class_id):
        self._class_id = class_id
        score_config = messages_pb2.PerplexityScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.class_id = class_id
        reconfigure_score_in_master(self._master, score_config, self._name)

    @topic_names.setter
    def topic_names(self, topic_names):
        self._topic_names = topic_names
        score_config = messages_pb2.PerplexityScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.ClearField('topic_names')
        for topic_name in topic_names:
            score_config.topic_name.append(topic_name)
        reconfigure_score_in_master(self._master, score_config, self._name)

    @eps.setter
    def eps(self, eps):
        self._eps = eps
        score_config = messages_pb2.PerplexityScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.eps = eps
        reconfigure_score_in_master(self._master, score_config, self._name)

    @dictionary_name.setter
    def dictionary_name(self, dictionary_name):
        self._dictionary_name = dictionary_name
        score_config = messages_pb2.PerplexityScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.dictionary = dictionary_name
        reconfigure_score_in_master(self._master, score_config, self._name)

    @use_unigram_document_model.setter
    def use_unigram_document_model(self, use_unigram_document_model):
        self._use_unigram_document_model = use_unigram_document_model
        score_config = messages_pb2.PerplexityScoreConfig()
        score_config.CopyFrom(self._config)
        if use_unigram_document_model is True:
            score_config.model_type = library.PerplexityScoreConfig_Type_UnigramDocumentModel
        else:
            score_config.model_type = library.PerplexityScoreConfig_Type_UnigramCollectionModel
        reconfigure_score_in_master(self._master, score_config, self._name)


###################################################################################################
class ItemsProcessedScore(object):
    """ItemsProcessedScore is a score in ArtmModel (public class)

    Args:
      name (str): the identifier of score, will be auto-generated if not specified
    """

    def __init__(self, name=None):
        config = messages_pb2.ItemsProcessedScoreConfig()

        if name is None:
            name = 'PerplexityScore:' + uuid.uuid1().urn

        self._name = name
        self._config = config
        self._type = library.ScoreConfig_Type_ItemsProcessed
        self._model = None  # Reserve place for the model
        self._master = None  # Reserve place for the master (to reconfigure Scores)
        self._score = None  # Reserve place for the score

    @property
    def name(self):
        return self._name

    @property
    def config(self):
        return self._config

    @property
    def type(self):
        return self._type

    @property
    def model(self):
        return self._model

    @property
    def score(self):
        return self._score

    @property
    def master(self):
        return self._master


###################################################################################################
class TopTokensScore(object):
    """TopTokensScore is a score in ArtmModel (public class)

    Args:
      name (str): the identifier of score, will be auto-generated if not specified
      class_id (str): class_id to score, default=None
      topic_names (list of str): list of names of topics to regularize, will
      score all topics if not specified
      num_tokens (int): number of tokens with max probability
      in each topic, default=10
      dictionary_name (str): BigARTM collection dictionary, won't use dictionary
      if not specified
    """

    def __init__(self, name=None, class_id=None, topic_names=None,
                 num_tokens=None, dictionary_name=None):
        config = messages_pb2.TopTokensScoreConfig()
        self._class_id = '@default_class'
        self._topic_names = []
        self._num_tokens = 10
        self._dictionary_name = ''

        if name is None:
            name = 'TopTokensScore:' + uuid.uuid1().urn
        if class_id is not None:
            config.class_id = class_id
            self._class_id = class_id
        if topic_names is not None:
            config.ClearField('theta_sparsity_topic_name')
            for topic_name in topic_names:
                config.theta_sparsity_topic_name.append(topic_name)
                self._topic_names.append(topic_name)
        if num_tokens is not None:
            config.num_tokens = num_tokens
            self._num_tokens = num_tokens
        if dictionary_name is not None:
            self._dictionary_name = dictionary_name
            config.cooccurrence_dictionary_name = dictionary_name

        self._name = name
        self._config = config
        self._type = library.ScoreConfig_Type_TopTokens
        self._model = None  # Reserve place for the model
        self._master = None  # Reserve place for the master (to reconfigure Scores)
        self._score = None  # Reserve place for the score

    @property
    def name(self):
        return self._name

    @property
    def config(self):
        return self._config

    @property
    def type(self):
        return self._type

    @property
    def class_id(self):
        return self._class_id

    @property
    def topic_names(self):
        return self._topic_names

    @property
    def num_tokens(self):
        return self._num_tokens

    @property
    def dictionary_name(self):
        return self._dictionary_name

    @property
    def model(self):
        return self._model

    @property
    def score(self):
        return self._score

    @property
    def master(self):
        return self._master

    @class_id.setter
    def class_id(self, class_id):
        self._class_id = class_id
        score_config = messages_pb2.TopTokensScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.class_id = class_id
        reconfigure_score_in_master(self._master, score_config, self._name)

    @topic_names.setter
    def topic_names(self, topic_names):
        self._topic_names = topic_names
        score_config = messages_pb2.TopTokensScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.ClearField('topic_names')
        for topic_name in topic_names:
            score_config.topic_name.append(topic_name)
        reconfigure_score_in_master(self._master, score_config, self._name)

    @num_tokens.setter
    def num_tokens(self, num_tokens):
        self._num_tokens = num_tokens
        score_config = messages_pb2.TopTokensScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.num_tokens = num_tokens
        reconfigure_score_in_master(self._master, score_config, self._name)

    @dictionary_name.setter
    def dictionary_name(self, dictionary_name):
        self._dictionary_name = dictionary_name
        score_config = messages_pb2.TopTokensScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.cooccurrence_dictionary_name = dictionary_name
        reconfigure_score_in_master(self._master, score_config, self._name)


###################################################################################################
class ThetaSnippetScore(object):
    """ThetaSnippetScore is a score in ArtmModel (public class)

    Args:
      name (str): the identifier of score, will be auto-generated if not specified
      item_ids (list of int): list of names of items to show, default=None
      num_items (int): number of theta vectors to show from the
      beginning (no sense if item_ids given), default=10
    """

    def __init__(self, name=None, item_ids=None, num_items=None):
        config = messages_pb2.ThetaSnippetScoreConfig()
        self._item_ids = []
        self._num_items = 10

        if name is None:
            name = 'ThetaSnippetScore:' + uuid.uuid1().urn
        if item_ids is not None:
            config.ClearField('item_id')
            for item_id in item_ids:
                config.item_id.append(item_id)
                self._item_ids.append(item_id)
        if num_items is not None:
            config.item_count = num_items
            self._num_items = num_items

        self._name = name
        self._config = config
        self._type = library.ScoreConfig_Type_ThetaSnippet
        self._model = None  # Reserve place for the model
        self._master = None  # Reserve place for the master (to reconfigure Scores)
        self._score = None  # Reserve place for the score

    @property
    def name(self):
        return self._name

    @property
    def config(self):
        return self._config

    @property
    def type(self):
        return self._type

    @property
    def item_ids(self):
        return self._item_ids

    @property
    def num_items(self):
        return self._num_items

    @property
    def model(self):
        return self._model

    @property
    def score(self):
        return self._score

    @property
    def master(self):
        return self._master

    @item_ids.setter
    def item_ids(self, item_ids):
        self._item_ids = item_ids
        score_config = messages_pb2.ThetaSnippetScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.ClearField('item_id')
        for item_id in item_ids:
            score_config.item_id.append(item_id)
        reconfigure_score_in_master(self._master, score_config, self._name)

    @num_items.setter
    def num_items(self, num_items):
        self._num_items = num_items
        score_config = messages_pb2.ThetaSnippetScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.item_count = num_items
        reconfigure_score_in_master(self._master, score_config, self._name)


###################################################################################################
class TopicKernelScore(object):
    """TopicKernelScore is a score in ArtmModel (public class)

    Args:
      name (str): the identifier of score, will be auto-generated if not specified
      class_id (str): class_id to score, default=None
      topic_names (list of str): list of names of topics to regularize, will
      score all topics if not specified
      eps (double): the tolerance const, everything < eps
      considered to be zero, default=1e-37
      dictionary_name (str): BigARTM collection dictionary, won't use dictionary
      if not specified
      probability_mass_threshold (double): the threshold for p(t|w) values to
      get token into topic kernel. Should be in (0, 1), default=0.1
    """

    def __init__(self, name=None, class_id=None, topic_names=None, eps=None,
                 dictionary_name=None, probability_mass_threshold=None):
        config = messages_pb2.TopicKernelScoreConfig()
        self._class_id = '@default_class'
        self._topic_names = []
        self._eps = GLOB_EPS
        self._dictionary_name = ''
        self._probability_mass_threshold = 0.1

        if name is None:
            name = 'TopicKernelScore:' + uuid.uuid1().urn
        if class_id is not None:
            config.class_id = class_id
            self._class_id = class_id
        if topic_names is not None:
            config.ClearField('theta_sparsity_topic_name')
            for topic_name in topic_names:
                config.theta_sparsity_topic_name.append(topic_name)
                self._topic_names.append(topic_name)
        if eps is not None:
            config.theta_sparsity_eps = eps
            self._eps = eps
        if dictionary_name is not None:
            self._dictionary_name = dictionary_name
            config.cooccurrence_dictionary_name = dictionary_name
        if probability_mass_threshold is not None:
            config.probability_mass_threshold = probability_mass_threshold
            self._probability_mass_threshold = probability_mass_threshold

        self._name = name
        self._config = config
        self._type = library.ScoreConfig_Type_TopicKernel
        self._model = None  # Reserve place for the model
        self._master = None  # Reserve place for the master (to reconfigure Scores)
        self._score = None  # Reserve place for the score

    @property
    def name(self):
        return self._name

    @property
    def config(self):
        return self._config

    @property
    def type(self):
        return self._type

    @property
    def class_id(self):
        return self._class_id

    @property
    def topic_names(self):
        return self._topic_names

    @property
    def eps(self):
        return self._eps

    @property
    def dictionary_name(self):
        return self._dictionary_name

    @property
    def probability_mass_threshold(self):
        return self._probability_mass_threshold

    @property
    def model(self):
        return self._model

    @property
    def score(self):
        return self._score

    @property
    def master(self):
        return self._master

    @class_id.setter
    def class_id(self, class_id):
        self._class_id = class_id
        score_config = messages_pb2.TopicKernelScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.class_id = class_id
        reconfigure_score_in_master(self._master, score_config, self._name)

    @topic_names.setter
    def topic_names(self, topic_names):
        self._topic_names = topic_names
        score_config = messages_pb2.TopicKernelScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.ClearField('topic_names')
        for topic_name in topic_names:
            score_config.topic_name.append(topic_name)
        reconfigure_score_in_master(self._master, score_config, self._name)

    @eps.setter
    def eps(self, eps):
        self._eps = eps
        score_config = messages_pb2.TopicKernelScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.eps = eps
        reconfigure_score_in_master(self._master, score_config, self._name)

    @dictionary_name.setter
    def dictionary_name(self, dictionary_name):
        self._dictionary_name = dictionary_name
        score_config = messages_pb2.TopicKernelScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.cooccurrence_dictionary_name = dictionary_name
        reconfigure_score_in_master(self._master, score_config, self._name)

    @probability_mass_threshold.setter
    def probability_mass_threshold(self, probability_mass_threshold):
        self._probability_mass_threshold = probability_mass_threshold
        score_config = messages_pb2.TopicKernelScoreConfig()
        score_config.CopyFrom(self._config)
        score_config.probability_mass_threshold = probability_mass_threshold
        reconfigure_score_in_master(self._master, score_config, self._name)


###################################################################################################
# SECTION OF SCORE INFO CLASSES
###################################################################################################
class SparsityPhiScoreInfo(object):
    """SparsityPhiScoreInfo represents a result of counting
    SparsityPhiScore (private class)

    Args:
      score (reference): reference to Score object, no default
    """

    def __init__(self, score):
        self._name = score.name
        self._value = []
        self._zero_tokens = []
        self._total_tokens = []

    def add(self, score=None):
        """SparsityPhiScoreInfo.add() --- add info about score after synchronization

        Args:
          score (reference): reference to score object, if not specified
          means 'Add None values'
        """
        if score is not None:
            _data = score.score.GetValue(score._model)

            self._value.append(_data.value)
            self._zero_tokens.append(_data.zero_tokens)
            self._total_tokens.append(_data.total_tokens)
        else:
            self._value.append(None)
            self._zero_tokens.append(None)
            self._total_tokens.append(None)

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        """Returns:
          list of double: value of Phi sparsity on synchronizations
        """
        return self._value

    @property
    def zero_tokens(self):
        """Returns:
          list of int: number of zero rows in Phi on synchronizations
        """
        return self._zero_tokens

    @property
    def total_tokens(self):
        """Returns:
          list of int: total number of rows in Phi on synchronizations
        """
        return self._total_tokens

    @property
    def last_value(self):
        """Returns:
        double: value of Phi sparsity on the last synchronization
        """
        return self._value[len(self._value) - 1]

    @property
    def last_zero_tokens(self):
        """Returns:
        int: number of zero rows in Phi on the last synchronization
        """
        return self._zero_tokens[len(self._zero_tokens) - 1]

    @property
    def last_total_tokens(self):
        """Returns:
        int: total number of rows in Phi on the last synchronization
        """
        return self._total_tokens[len(self._total_tokens) - 1]


###################################################################################################
class SparsityThetaScoreInfo(object):
    """SparsityThetaScoreInfo represents a result of counting
    SparsityThetaScore (private class)

    Args:
      score (reference): reference to Score object, no default
    """

    def __init__(self, score):
        self._name = score.name
        self._value = []
        self._zero_topics = []
        self._total_topics = []

    def add(self, score=None):
        """SparsityThetaScoreInfo.add() --- add info about score
        after synchronization

        Args:
          score (reference): reference to score object, if not specified
          means 'Add None values'
        """
        if score is not None:
            _data = score.score.GetValue(score._model)

            self._value.append(_data.value)
            self._zero_topics.append(_data.zero_topics)
            self._total_topics.append(_data.total_topics)
        else:
            self._value.append(None)
            self._zero_topics.append(None)
            self._total_topics.append(None)

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        """Returns:
          list of double: value of Theta sparsity on synchronizations
        """
        return self._value

    @property
    def zero_topics(self):
        """Returns:
          list of int: number of zero rows in Theta on synchronizations
        """
        return self._zero_topics

    @property
    def total_topics(self):
        """Returns:
          list of int: total number of rows in Theta on synchronizations
        """
        return self._total_topics

    @property
    def last_value(self):
        """Returns:
          double: value of Theta sparsity on the last synchronization
        """
        return self._value[len(self._value) - 1]

    @property
    def last_zero_topics(self):
        """Returns:
          int: number of zero rows in Theta on the last synchronization
        """
        return self._zero_topics[len(self._zero_topics) - 1]

    @property
    def last_total_topics(self):
        """Returns:
          int: total number of rows in Theta on the last synchronization
        """
        return self._total_topics[len(self._total_topics) - 1]


###################################################################################################
class PerplexityScoreInfo(object):
    """PerplexityScoreInfo represents a result of counting PerplexityScore
    (private class)

    Args:
      score (reference): reference to Score object, no default
    """

    def __init__(self, score):
        self._name = score.name
        self._value = []
        self._raw = []
        self._normalizer = []
        self._zero_tokens = []
        self._theta_sparsity_value = []
        self._theta_sparsity_zero_topics = []
        self._theta_sparsity_total_topics = []

    def add(self, score=None):
        """PerplexityScoreInfo.add() --- add info about score after
        synchronization

        Args:
          score (reference): reference to score object, if not specified
          means 'Add None values'
        """
        if score is not None:
            _data = score.score.GetValue(score._model)

            self._value.append(_data.value)
            self._raw.append(_data.raw)
            self._normalizer.append(_data.normalizer)
            self._zero_tokens.append(_data.zero_words)
            self._theta_sparsity_value.append(_data.theta_sparsity_value)
            self._theta_sparsity_zero_topics.append(_data.theta_sparsity_zero_topics)
            self._theta_sparsity_total_topics.append(_data.theta_sparsity_total_topics)
        else:
            self._value.append(None)
            self._raw.append(None)
            self._normalizer.append(None)
            self._zero_tokens.append(None)
            self._theta_sparsity_value.append(None)
            self._theta_sparsity_zero_topics.append(None)
            self._theta_sparsity_total_topics.append(None)

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        """Returns:
          list of double: value of perplexity on synchronizations
        """
        return self._value

    @property
    def raw(self):
        """Returns:
          list of double: raw value in formula of perplexity on synchronizations
        """
        return self._raw

    @property
    def normalizer(self):
        """Returns:
          list double: normalizer value in formula of perplexity on synchronizations
        """
        return self._normalizer

    @property
    def zero_tokens(self):
        """Returns:
          list of int: number of tokens with zero counters on synchronizations
        """
        return self._zero_tokens

    @property
    def theta_sparsity_value(self):
        """Returns:
          list of double: Theta sparsity value on synchronizations
        """
        return self._theta_sparsity_value

    @property
    def theta_sparsity_zero_topics(self):
        """Returns:
        list of int: number of zero rows in Theta on synchronizations
        """
        return self._theta_sparsity_zero_topics

    @property
    def theta_sparsity_total_topics(self):
        """Returns:
          list of int: total number of rows in Theta on synchronizations
        """
        return self._theta_sparsity_total_topics

    @property
    def last_value(self):
        """Returns:
          double: value of perplexity on the last synchronization
        """
        return self._value[len(self._value) - 1]

    @property
    def last_raw(self):
        """Returns:
          double: raw value in formula of perplexity on the last synchronization
        """
        return self._raw[len(self._raw) - 1]

    @property
    def last_normalizer(self):
        """Returns:
          double: normalizer value in formula of perplexity on the last synchronization
        """
        return self._normalizer[len(self._normalizer) - 1]

    @property
    def last_zero_tokens(self):
        """Returns:
          int: number of tokens with zero counters on the last synchronization
        """
        return self._zero_tokens[len(self._zero_tokens) - 1]

    @property
    def last_theta_sparsity_value(self):
        """Returns:
          double: Theta sparsity value on the last synchronization
        """
        return self._theta_sparsity_value[len(self._theta_sparsity_value) - 1]

    @property
    def last_theta_sparsity_zero_topics(self):
        """Returns:
          int: number of zero rows in Theta on the last synchronization
        """
        return self._theta_sparsity_zero_topics[len(self._theta_sparsity_zero_topics) - 1]

    @property
    def last_theta_sparsity_total_topics(self):
        """Returns:
          int: total number of rows in Theta on the last synchronization
        """
        return self._theta_sparsity_total_topics[len(self._theta_sparsity_total_topics) - 1]


###################################################################################################
class ItemsProcessedScoreInfo(object):
    """ItemsProcessedScoreInfo represents a result of counting
    ItemsProcessedScore (private class)

    Args:
      score (reference): reference to Score object, no default
    """

    def __init__(self, score):
        self._name = score.name
        self._value = []

    def add(self, score=None):
        """ItemsProcessedScoreInfo.add() --- add info about score
        after synchronization

        Args:
          score (reference): reference to score object, if not specified
          means 'Add None values'
        """
        if score is not None:
            _data = score.score.GetValue(score._model)
            self._value.append(_data.value)
        else:
            self._value.append(None)

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        """Returns:
          list of int: total number of processed documents on synchronizations
        """
        return self._value

    @property
    def last_value(self):
        """Returns:
          int: total number of processed documents on the last synchronization
        """
        return self._value[len(self._value) - 1]


###################################################################################################
class TopTokensScoreInfo(object):
    """TopTokensScoreInfo represents a result of counting TopTokensScore
    (private class)

    Args:
      score (reference): reference to Score object, no default
    """

    def __init__(self, score):
        self._name = score.name
        self._num_tokens = []
        self._topic_info = []
        self._average_coherence = []

    def add(self, score=None):
        """TopTokensScoreInfo.add() --- add info about score
        after synchronization

        Args:
          score (reference): reference to score object, if not specified
          means 'Add None values'
        """
        if score is not None:
            _data = score.score.GetValue(score._model)

            self._num_tokens.append(_data.num_entries)

            self._topic_info.append({})
            index = len(self._topic_info) - 1
            topic_index = -1
            for topic_name in list(collections.OrderedDict.fromkeys(_data.topic_name)):
                topic_index += 1
                tokens = []
                weights = []
                for i in xrange(_data.num_entries):
                    if _data.topic_name[i] == topic_name:
                        tokens.append(_data.token[i])
                        weights.append(_data.weight[i])
                coherence = -1
                if len(_data.coherence.value) > 0:
                    coherence = _data.coherence.value[topic_index]
                self._topic_info[index][topic_name] = \
                    collections.namedtuple('TopTokensScoreTuple',
                                           ['tokens', 'weights', 'coherence'])
                self._topic_info[index][topic_name].tokens = tokens
                self._topic_info[index][topic_name].weights = weights
                self._topic_info[index][topic_name].coherence = coherence

            self._average_coherence.append(_data.average_coherence)
        else:
            self._num_tokens.append(None)
            self._topic_info.append(None)
            self._average_coherence.append(None)

    @property
    def name(self):
        return self._name

    @property
    def num_tokens(self):
        """Returns:
          list of int: reqested number of top tokens in each topic on
        synchronizations
        """
        return self._num_tokens

    @property
    def topic_info(self):
        """Returns:
          list of sets: information about top tokens per topic on synchronizations;
          each set contains information about topics,
          key --- name of topic, value --- named tuple:
          - *.topic_info[sync_index][topic_name].tokens --- list of top tokens
            for this topic
          - *.topic_info[sync_index][topic_name].weights --- list of weights
            (probabilities), corresponds the tokens
          - *.topic_info[sync_index][topic_name].coherence --- the coherency
            of topic due to it's top tokens
        """
        return self._topic_info

    @property
    def average_coherence(self):
        """Returns:
          list of double: average coherence of top tokens in all requested topics
          on synchronizations
        """
        return self._average_coherence

    @property
    def last_num_tokens(self):
        """Returns:
          int: reqested number of top tokens in each topic on the last
          synchronization
        """
        return self._num_tokens[len(self._num_tokens) - 1]

    @property
    def last_topic_info(self):
        """Returns:
          set: information about top tokens per topic on the last
          synchronization;
          each set contains information about topics,
          key --- name of topic, value --- named tuple:
          - *.last_topic_info[topic_name].tokens --- list of top tokens
            for this topic
          - *.last_topic_info[topic_name].weights --- list of weights
            (probabilities), corresponds the tokens
          - *.last_topic_info[topic_name].coherence --- the coherency
            of topic due to it's top tokens
        """
        return self._topic_info[len(self._topic_info) - 1]

    @property
    def last_average_coherence(self):
        """Returns:
          double: average coherence of top tokens in all requested topics
          on the last synchronization
        """
        return self._average_coherence[len(self._average_coherence) - 1]


###################################################################################################
class TopicKernelScoreInfo(object):
    """TopicKernelScoreInfo represents a result of counting TopicKernelScore
    (private class)

    Args:
      score (reference): reference to Score object, no default
    """

    def __init__(self, score):
        self._name = score.name
        self._topic_info = []
        self._average_coherence = []
        self._average_size = []
        self._average_contrast = []
        self._average_purity = []

    def add(self, score=None):
        """TopicKernelScoreInfo.add() --- add info about score after
        synchronization

        Args:
          score (reference): reference to score object, if not specified
          means 'Add None values'
        """
        if score is not None:
            _data = score.score.GetValue(score._model)

            self._topic_info.append({})
            index = len(self._topic_info) - 1
            topic_index = -1
            for topic_name in _data.topic_name.value:
                topic_index += 1
                tokens = [token for token in _data.kernel_tokens[topic_index].value]
                coherence = -1
                if len(_data.coherence.value) > 0:
                    coherence = _data.coherence.value[topic_index]
                self._topic_info[index][topic_name] = \
                    collections.namedtuple('TopicKernelScoreTuple',
                                           ['tokens', 'size', 'contrast', 'purity', 'coherence'])
                self._topic_info[index][topic_name].tokens = tokens
                self._topic_info[index][topic_name].size = _data.kernel_size.value[topic_index]
                self._topic_info[index][topic_name].contrast = \
                    _data.kernel_purity.value[topic_index]
                self._topic_info[index][topic_name].purity = \
                    _data.kernel_contrast.value[topic_index]
                self._topic_info[index][topic_name].coherence = coherence

            self._average_coherence.append(_data.average_coherence)
            self._average_size.append(_data.average_kernel_size)
            self._average_contrast.append(_data.average_kernel_contrast)
            self._average_purity.append(_data.average_kernel_purity)
        else:
            self._topic_info.append(None)
            self._average_coherence.append(None)
            self._average_size.append(None)
            self._average_contrast.append(None)
            self._average_purity.append(None)

    @property
    def name(self):
        return self._name

    @property
    def topic_info(self):
        """Returns:
          list of sets: information about kernel tokens per topic on
          synchronizations; each set contains information
          about topics, key --- name of topic, value --- named tuple:
          - *.topic_info[sync_index][topic_name].tokens --- list of
            kernel tokens for this topic
          - *.topic_info[sync_index][topic_name].size --- size of
            kernel for this topic
          - *.topic_info[sync_index][topic_name].contrast --- contrast of
            kernel for this topic.
          - *.topic_info[sync_index][topic_name].purity --- purity of kernel
            for this topic
          - *.topic_info[sync_index][topic_name].coherence --- the coherency of
            topic due to it's kernel
        """
        return self._topic_info

    @property
    def average_coherence(self):
        """Returns:
          list of double: average coherence of kernel tokens in all requested
          topics on synchronizations
        """
        return self._average_coherence

    @property
    def average_size(self):
        """Returns:
          list of double: average kernel size of all requested topics on
          synchronizations
        """
        return self._average_size

    @property
    def average_contrast(self):
        """Returns:
          list of double: average kernel contrast of all requested topics on
        synchronizations
        """
        return self._average_contrast

    @property
    def average_purity(self):
        """Returns:
          list of double: average kernel purity of all requested topics on
        synchronizations
        """
        return self._average_purity

    @property
    def last_topic_info(self):
        """Returns:
          set: information about kernel tokens per topic on the last
          synchronization; each set contains information about topics,
          key --- name of topic, value --- named tuple:
          - *.topic_info[topic_name].tokens --- list of
            kernel tokens for this topic
          - *.topic_info[topic_name].size --- size of
            kernel for this topic
          - *.topic_info[topic_name].contrast --- contrast of
            kernel for this topic
          - *.topic_info[topic_name].purity --- purity of kernel
            for this topic
          - *.topic_info[topic_name].coherence --- the coherency of
            topic due to it's kernel
        """
        return self._topic_info[len(self._topic_info) - 1]

    @property
    def last_average_coherence(self):
        """Returns:
          double: average coherence of kernel tokens in all requested
          topics on the last synchronization
        """
        return self._average_coherence[len(self._average_coherence) - 1]

    @property
    def last_average_size(self):
        """Returns:
          double: average kernel size of all requested topics on
          the last synchronization
        """
        return self._average_size[len(self._average_size) - 1]

    @property
    def last_average_contrast(self):
        """Returns:
          double: average kernel contrast of all requested topics on
          the last synchronization
        """
        return self._average_contrast[len(self._average_contrast) - 1]

    @property
    def last_average_purity(self):
        """Returns:
          double: average kernel purity of all requested topics on
          the last synchronization
        """
        return self._average_purity[len(self._average_purity) - 1]


###################################################################################################
class ThetaSnippetScoreInfo(object):
    """ThetaSnippetScoreInfo represents a result of counting
    ThetaSnippetScore (private class)

    Args:
      score (reference): reference to Score object, no default
    """

    def __init__(self, score):
        self._name = score.name
        self._document_ids = []
        self._snippet = []

    def add(self, score=None):
        """ThetaSnippetScoreInfo.add() --- add info about score after
        synchronization

        Args:
          score (reference): reference to score object, if not specified
          means 'Add None values'
        """
        if score is not None:
            _data = score.score.GetValue(score._model)

            self._document_ids.append([item_id for item_id in _data.item_id])
            self._snippet.append(
                [[theta_td for theta_td in theta_d.value] for theta_d in _data.values])
        else:
            self._document_ids.append(None)
            self._snippet.append(None)

    @property
    def name(self):
        return self._name

    @property
    def snippet(self):
        """Returns:
          list of lists of lists of double: the snippet (part) of Theta
          corresponds to documents from document_ids on each synchronizations;
          each most internal list --- theta_d vector for document d,
          in direct order of document_ids
        """
        return self._snippet

    @property
    def document_ids(self):
        """Returns:
          list of int: ids of documents in snippet on synchronizations
        """
        return self._document_ids

    @property
    def last_snippet(self):
        """Returns:
          list of lists of double: the snippet (part) of Theta corresponds
          to documents from document_ids on last synchronization;
          each internal list --- theta_d vector for document d,
          in direct order of document_ids
        """
        return self._snippet

    @property
    def last_document_ids(self):
        """Returns:
          list of int: ids of documents in snippet on the last synchronization
        """
        return self._document_ids


###################################################################################################
# SECTION OF ARTM MODEL CLASS
###################################################################################################
class ArtmModel(object):
    """ArtmModel represents a topic model (public class)

    Args:
      num_processors (int): how many threads will be used for model training,
      if not specified then number of threads will be detected by the library
      topic_names (list of str): names of topics in model, if not specified will be
      auto-generated by library according to num_topics
      num_topics (int): number of topics in model (is used if topic_names
      not specified), default=10
      class_ids (dict): list of class_ids and their weights to be used in model,
      key --- class_id, value --- weight, if not specified then all class_ids
      will be used
      num_document_passes (int): number of iterations over each document
      during processing, default=10
      cache_theta (bool): save or not the Theta matrix in model. Necessary
      if ArtmModel.get_theta() usage expects, default=True

    Important public fields:
      regularizers: contains dict of regularizers, included into model
      scores: contains dict of scores, included into model
      scores_info: contains dict of scoring results;
      key --- score name, value --- ScoreInfo object, which contains info about
      values of score on each synchronization in list

    NOTE:
      - Here and anywhere in BigARTM empty topic_names or class_ids means that
      model (or regularizer, or score) should use all topics or class_ids.
      - If some fields of regularizers or scores are not defined by
      user --- internal library defaults would be used.
      - If field 'topics_name' == [], it will be generated by BigARTM and will
      be available using ArtmModel.topics_name().
    """

    # ========== CONSTRUCTOR ==========
    def __init__(self, num_processors=0, topic_names=None, num_topics=10,
                 class_ids=None, num_document_passes=10, cache_theta=True):
        self._num_processors = 0
        self._num_topics = 10
        self._num_document_passes = 10
        self._cache_theta = True

        if topic_names is None or topic_names is []:
            self._topic_names = []
            if num_topics > 0:
                self._num_topics = num_topics
        else:
            self._topic_names = topic_names
            self._num_topics = len(topic_names)

        if class_ids is None:
            self._class_ids = {}
        elif len(class_ids) > 0:
            self._class_ids = class_ids

        if num_processors > 0:
            self._num_processors = num_processors

        if num_document_passes > 0:
            self._num_document_passes = num_document_passes

        if isinstance(cache_theta, bool):
            self._cache_theta = cache_theta

        self._master = library.MasterComponent()
        self._master.config().processors_count = self._num_processors
        self._master.config().cache_theta = cache_theta
        self._master.Reconfigure()

        self._model = 'pwt'
        self._regularizers = Regularizers(self._master)
        self._scores = Scores(self._master, self._model)

        self._scores_info = {}
        self._synchronizations_processed = 0
        self._initialized = False

    # ========== PROPERTIES ==========
    @property
    def num_processors(self):
        return self._num_processors

    @property
    def num_document_passes(self):
        return self._num_document_passes

    @property
    def cache_theta(self):
        return self._cache_theta

    @property
    def num_topics(self):
        return self._num_topics

    @property
    def topic_names(self):
        return self._topic_names

    @property
    def class_ids(self):
        return self._class_ids

    @property
    def regularizers(self):
        return self._regularizers

    @property
    def scores(self):
        return self._scores

    @property
    def scores_info(self):
        return self._scores_info

    @property
    def master(self):
        return self._master

    @property
    def num_phi_updates(self):
        return self._synchronizations_processed

    # ========== SETTERS ==========
    @num_processors.setter
    def num_processors(self, num_processors):
        if num_processors <= 0 or not isinstance(num_processors, int):
            raise IOError('Number of processors should be a positive integer')
        else:
            self._num_processors = num_processors
            self._master.config().processors_count = num_processors
            self._master.Reconfigure()

    @num_document_passes.setter
    def num_document_passes(self, num_document_passes):
        if num_document_passes <= 0 or not isinstance(num_document_passes, int):
            raise IOError("Number of passes through documents" +
                          "should be a positive integer")
        else:
            self._num_document_passes = num_document_passes

    @cache_theta.setter
    def cache_theta(self, cache_theta):
        if not isinstance(cache_theta, bool):
            raise IOError('cache_theta should be bool')
        else:
            self._cache_theta = cache_theta
            self._master.config().cache_theta = cache_theta
            self._master.Reconfigure()

    @num_topics.setter
    def num_topics(self, num_topics):
        if num_topics <= 0 or not isinstance(num_topics, int):
            raise IOError('Number of topics should be a positive integer')
        else:
            self._num_topics = num_topics

    @topic_names.setter
    def topic_names(self, topic_names):
        if len(topic_names) < 0:
            raise IOError('Number of topic names should be non-negative')
        else:
            self._topic_names = topic_names
            self._num_topics = len(topic_names)

    @class_ids.setter
    def class_ids(self, class_ids):
        if len(class_ids) < 0:
            raise IOError('Number of (class_id, class_weight) pairs should be non-negative')
        else:
            self._class_ids = class_ids

        # ========== METHODS ==========

    def load_dictionary(self, dictionary_name=None, dictionary_path=None):
        """ArtmModel.load_dictionary() --- load the BigARTM dictionary of
        the collection into the library

        Args:
          dictionary_name (str): the name of the dictionary in the library, default=None
          dictionary_path (str): full file name of the dictionary, default=None
        """
        if dictionary_path is not None and dictionary_name is not None:
            self._master.ImportDictionary(dictionary_name, dictionary_path)
        elif dictionary_path is None:
            raise IOError('ArtmModel.load_dictionary(): dictionary_path is None')
        else:
            raise IOError('ArtmModel.load_dictionary(): dictionary_name is None')

    def remove_dictionary(self, dictionary_name=None):
        """ArtmModel.remove_dictionary() --- remove the loaded BigARTM dictionary
        from the library

        Args:
          dictionary_name (str): the name of the dictionary in th library, default=None
        """
        if dictionary_name is not None:
            self._master.lib_.ArtmDisposeDictionary(self._master.id_, dictionary_name)
        else:
            raise IOError('ArtmModel.remove_dictionary(): dictionary_name is None')

    def fit_offline(self, collection_name=None, batches=None, data_path='',
                    num_collection_passes=1, decay_weight=0.0, apply_weight=1.0,
                    reset_theta_scores=False, data_format='batches', batch_size=1000):
        """ArtmModel.fit_offline() --- proceed the learning of
        topic model in off-line mode

        Args:
          collection_name (str): the name of text collection (required if
          data_format == 'bow_uci'), default=None
          batches (list of str): list of file names of batches to be processed.
          If not None, than data_format should be 'batches'. Format --- '*.batch',
          default=None
          data_path (str):
          1) if data_format == 'batches' => folder containing batches and dictionary;
          2) if data_format == 'bow_uci' => folder containing
            docword.collection_name.txt and vocab.collection_name.txt files;
          3) if data_format == 'vowpal_wabbit' => file in Vowpal Wabbit format;
          4) if data_format == 'plain_text' => file with text;
          default=''
          num_collection_passes (int): number of iterations over whole given
          collection, default=1
          decay_weight (int): coefficient for applying old n_wt counters,
          default=0.0 (apply_weight + decay_weight = 1.0)
          apply_weight (int): coefficient for applying new n_wt counters,
          default=1.0 (apply_weight + decay_weight = 1.0)
          reset_theta_scores (bool): reset accumulated Theta scores
          before learning, default=False
          data_format (str): the type of input data;
          1) 'batches' --- the data in format of BigARTM;
          2) 'bow_uci' --- Bag-Of-Words in UCI format;
          3) 'vowpal_wabbit' --- Vowpal Wabbit format;
          4) 'plain_text' --- source text;
          default='batches'

          Next argument has sense only if data_format is not 'batches'
          (e.g. parsing is necessary).
            batch_size (int): number of documents to be stored ineach batch,
            default=1000

        Note:
          ArtmModel.initialize() should be proceed before first call
          ArtmModel.fit_offline(), or it will be initialized by dictionary
          during first call.
        """
        if collection_name is None and data_format == 'bow_uci':
            raise IOError('ArtmModel.fit_offline(): No collection name was given')

        if not data_format == 'batches' and batches is not None:
            raise IOError("ArtmModel.fit_offline(): batches != None" +
                          "require data_format == batches")

        target_folder = data_path
        batches_list = []
        if data_format == 'batches':
            if batches is None:
                batches_list = glob.glob(data_path + '/*.batch')
                if len(batches_list) < 1:
                    raise RuntimeError('ArtmModel.fit_offline(): No batches were found')
            else:
                batches_list = [data_path + '/' + batch for batch in batches]

        elif data_format == 'bow_uci' or data_format == 'vowpal_wabbit':
            target_folder = data_path + '/batches_temp_' + str(random.uniform(0, 1))
            collection_parser_config = create_parser_config(data_path,
                                                            collection_name,
                                                            target_folder,
                                                            batch_size,
                                                            data_format)
            library.Library().ParseCollection(collection_parser_config)
            batches_list = glob.glob(target_folder + '/*.batch')

        elif data_format == 'plain_text':
            raise NotImplementedError()
        else:
            raise IOError('ArtmModel.fit_offline(): Unknown data format')

        if not self._initialized:
            dictionary_name = 'dictionary' + str(uuid.uuid4())
            self._master.ImportDictionary(dictionary_name,
                                          os.path.join(target_folder, 'dictionary'))
            self.initialize(dictionary_name)
            self.remove_dictionary(dictionary_name)

        theta_regularizers, phi_regularizers = {}, {}
        for name, config in self._regularizers.data.iteritems():
            if config.type == THETA_REGULARIZER_TYPE:
                theta_regularizers[name] = config.tau
            else:
                phi_regularizers[name] = config.tau

        for _ in xrange(num_collection_passes):
            self._master.ProcessBatches(pwt=self._model,
                                        batches=batches_list,
                                        target_nwt='nwt_hat',
                                        regularizers=theta_regularizers,
                                        inner_iterations_count=self._num_document_passes,
                                        class_ids=self._class_ids,
                                        reset_scores=reset_theta_scores)
            self._synchronizations_processed += 1
            if self._synchronizations_processed == 1:
                self._master.MergeModel({self._model: decay_weight, 'nwt_hat': apply_weight},
                                        target_nwt='nwt', topic_names=self._topic_names)
            else:
                self._master.MergeModel({'nwt': decay_weight, 'nwt_hat': apply_weight},
                                        target_nwt='nwt', topic_names=self._topic_names)

            self._master.RegularizeModel(self._model, 'nwt', 'rwt', phi_regularizers)
            self._master.NormalizeModel('nwt', self._model, 'rwt')

            for name in self.scores.data.keys():
                if name not in self.scores_info:
                    if self.scores[name].type == library.ScoreConfig_Type_SparsityPhi:
                        self._scores_info[name] = SparsityPhiScoreInfo(self.scores[name])
                    elif self.scores[name].type == library.ScoreConfig_Type_SparsityTheta:
                        self._scores_info[name] = SparsityThetaScoreInfo(self.scores[name])
                    elif self.scores[name].type == library.ScoreConfig_Type_Perplexity:
                        self._scores_info[name] = PerplexityScoreInfo(self.scores[name])
                    elif self.scores[name].type == library.ScoreConfig_Type_ThetaSnippet:
                        self._scores_info[name] = ThetaSnippetScoreInfo(self.scores[name])
                    elif self.scores[name].type == library.ScoreConfig_Type_ItemsProcessed:
                        self._scores_info[name] = ItemsProcessedScoreInfo(self.scores[name])
                    elif self.scores[name].type == library.ScoreConfig_Type_TopTokens:
                        self._scores_info[name] = TopTokensScoreInfo(self.scores[name])
                    elif self.scores[name].type == library.ScoreConfig_Type_TopicKernel:
                        self._scores_info[name] = TopicKernelScoreInfo(self.scores[name])

                    for _ in xrange(self._synchronizations_processed - 1):
                        self._scores_info[name].add()

                self._scores_info[name].add(self.scores[name])

        # Remove temp batches folder if it necessary
        if not data_format == 'batches':
            shutil.rmtree(target_folder)

    def fit_online(self, collection_name=None, batches=None, data_path='',
                   tau0=1024.0, kappa=0.7, update_every=1, reset_theta_scores=False,
                   data_format='batches', batch_size=1000):
        """ArtmModel.fit_online() --- proceed the learning of topic model
        in on-line mode

        Args:
          collection_name (str): the name of text collection (required if
          data_format == 'bow_uci'), default=None
          batches (list of str): list of file names of batches to be processed.
          If not None, than data_format should be 'batches'. Format --- '*.batch',
          default=None
          data_path (str):
          1) if data_format == 'batches' => folder containing batches and dictionary;
          2) if data_format == 'bow_uci' => folder containing
            docword.collection_name.txt and vocab.collection_name.txt files;
          3) if data_format == 'vowpal_wabbit' => file in Vowpal Wabbit format;
          4) if data_format == 'plain_text' => file with text;
          default=''
          update_every (int): the number of batches; model will be updated once per it,
          default=1
          tau0 (float): coefficient (see kappa), default=1024.0
          kappa (float): power for tau0, default=0.7

          The formulas for decay_weight and apply_weight:
          update_count = current_processed_docs / (batch_size * update_every)
          rho = pow(tau0 + update_count, -kappa)
          decay_weight = 1-rho
          apply_weight = rho

          reset_theta_scores (bool): reset accumulated Theta scores
          before learning, default=False
          data_format (str): the type of input data;
          1) 'batches' --- the data in format of BigARTM;
          2) 'bow_uci' --- Bag-Of-Words in UCI format;
          3) 'vowpal_wabbit' --- Vowpal Wabbit format;
          4) 'plain_text' --- source text;
          default='batches'

          Next argument has sense only if data_format is not 'batches'
          (e.g. parsing is necessary).
            batch_size (int): number of documents to be stored ineach batch,
            default=1000

        Note:
          ArtmModel.initialize() should be proceed before first call
          ArtmModel.fit_online(), or it will be initialized by dictionary
          during first call.
        """
        if collection_name is None and data_format == 'bow_uci':
            raise IOError('ArtmModel.fit_online(): No collection name was given')

        if not data_format == 'batches' and batches is not None:
            raise IOError('batches != None require data_format == batches')

        target_folder = data_path
        batches_list = []
        if data_format == 'batches':
            if batches is None:
                batches_list = glob.glob(data_path + '/*.batch')
                if len(batches_list) < 1:
                    raise RuntimeError('ArtmModel.fit_online(): No batches were found')
            else:
                batches_list = [data_path + '/' + batch for batch in batches]

        elif data_format == 'bow_uci' or data_format == 'vowpal_wabbit':
            target_folder = data_path + '/batches_temp_' + str(random.uniform(0, 1))
            collection_parser_config = create_parser_config(data_path,
                                                            collection_name,
                                                            target_folder,
                                                            batch_size,
                                                            data_format)
            library.Library().ParseCollection(collection_parser_config)
            batches = glob.glob(target_folder + '/*.batch')

        elif data_format == 'plain_text':
            raise NotImplementedError()
        else:
            raise IOError('ArtmModel.fit_online(): Unknown data format')

        if not self._initialized:
            dictionary_name = 'dictionary' + str(uuid.uuid4())
            self._master.ImportDictionary(dictionary_name,
                                          os.path.join(target_folder, 'dictionary'))
            self.initialize(dictionary_name)
            self.remove_dictionary(dictionary_name)

        theta_regularizers, phi_regularizers = {}, {}
        for name, config in self._regularizers.data.iteritems():
            if config.type == THETA_REGULARIZER_TYPE:
                theta_regularizers[name] = config.tau
            else:
                phi_regularizers[name] = config.tau

        batches_to_process = []
        current_processed_documents = 0
        for batch_index, batch_filename in enumerate(batches_list):
            batches_to_process.append(batch_filename)
            if ((batch_index + 1) % update_every == 0) or ((batch_index + 1) == len(batches_list)):
                self._master.ProcessBatches(pwt=self._model,
                                            batches=batches_to_process,
                                            target_nwt='nwt_hat',
                                            regularizers=theta_regularizers,
                                            inner_iterations_count=self._num_document_passes,
                                            class_ids=self._class_ids,
                                            reset_scores=reset_theta_scores)

                current_processed_documents += batch_size * update_every
                update_count = current_processed_documents / (batch_size * update_every)
                rho = pow(tau0 + update_count, -kappa)
                decay_weight, apply_weight = 1 - rho, rho

                self._synchronizations_processed += 1
                if self._synchronizations_processed == 1:
                    self._master.MergeModel({self._model: decay_weight, 'nwt_hat': apply_weight},
                                            target_nwt='nwt', topic_names=self._topic_names)
                else:
                    self._master.MergeModel({'nwt': decay_weight, 'nwt_hat': apply_weight},
                                            target_nwt='nwt', topic_names=self._topic_names)

                self._master.RegularizeModel(self._model, 'nwt', 'rwt', phi_regularizers)
                self._master.NormalizeModel('nwt', self._model, 'rwt')
                batches_to_process = []

                for name in self.scores.data.keys():
                    if name not in self.scores_info:
                        if self.scores[name].type == library.ScoreConfig_Type_SparsityPhi:
                            self._scores_info[name] = SparsityPhiScoreInfo(self.scores[name])
                        elif self.scores[name].type == library.ScoreConfig_Type_SparsityTheta:
                            self._scores_info[name] = SparsityThetaScoreInfo(self.scores[name])
                        elif self.scores[name].type == library.ScoreConfig_Type_Perplexity:
                            self._scores_info[name] = PerplexityScoreInfo(self.scores[name])
                        elif self.scores[name].type == library.ScoreConfig_Type_ThetaSnippet:
                            self._scores_info[name] = ThetaSnippetScoreInfo(self.scores[name])
                        elif self.scores[name].type == library.ScoreConfig_Type_ItemsProcessed:
                            self._scores_info[name] = ItemsProcessedScoreInfo(self.scores[name])
                        elif self.scores[name].type == library.ScoreConfig_Type_TopTokens:
                            self._scores_info[name] = TopTokensScoreInfo(self.scores[name])
                        elif self.scores[name].type == library.ScoreConfig_Type_TopicKernel:
                            self._scores_info[name] = TopicKernelScoreInfo(self.scores[name])

                        for _ in xrange(self._synchronizations_processed - 1):
                            self._scores_info[name].add()

                    self._scores_info[name].add(self.scores[name])

        # Remove temp batches folder if it necessary
        if not data_format == 'batches':
            shutil.rmtree(target_folder)

    def save(self, file_name='artm_model'):
        """ArtmModel.save() --- save the topic model to disk

        Args:
          file_name (str): the name of file to store model, default='artm_model'
        """
        if not self._initialized:
            raise RuntimeError("Model does not exist yet. Use " +
                               "ArtmModel.initialize()/ArtmModel.fit_*()")

        if os.path.isfile(file_name):
            os.remove(file_name)
        self._master.ExportModel(self._model, file_name)

    def load(self, file_name):
        """ArtmModel.load() --- load the topic model,
        saved by ArtmModel.save(), from disk

        Args:
          file_name (str) --- the name of file containing model, no default

        Note:
          Loaded model will overwrite ArtmModel.topic_names and
          ArtmModel.num_topics fields. Also it will empty
          ArtmModel.scores_info.
        """
        self._master.ImportModel(self._model, file_name)
        self._initialized = True
        topic_model = self._master.GetTopicModel(model=self._model, use_matrix=False)
        self._topic_names = [topic_name for topic_name in topic_model.topic_name]
        self._num_topics = topic_model.topics_count

        # Remove all info about previous iterations
        self._scores_info = {}
        self._synchronizations_processed = 0

    def to_csv(self, file_name='artm_model.csv'):
        """ArtmModel.to_csv() --- save the topic model to disk in
        .csv format (can't be loaded back)

        Args:
          file_name (str): the name of file to store model, default='artm_model.csv'
        """
        if not self._initialized:
            raise RuntimeError("Model does not exist yet. Use " +
                               "ArtmModel.initialize()/ArtmModel.fit_*()")

        if os.path.isfile(file_name):
            os.remove(file_name)

        with open(file_name, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar=';', quoting=csv.QUOTE_MINIMAL)
            if len(self._topic_names) > 0:
                writer.writerow(['Token'] + ['Class ID'] +
                                ['TOPIC: ' + topic_name for topic_name in self._topic_names])

            model = self._master.GetTopicModel(model=self._model)
            index = -1
            for row in model[1]:
                index += 1
                writer.writerow(
                    [model[0].token[index]] + [model[0].class_id[index]] +
                    [round(token_w, 5) for token_w in row])

    def get_phi(self, topic_names=None, class_ids=None):
        """ArtmModel.get_phi() --- get Phi matrix of model

        Args:
          topic_names (list of str): list with topics to extract,
          default=None (means all topics)
          class_ids (list of str): list with class ids to extract,
          default=None (means all class ids)

        Returns:
          pandas.DataFrame: (data, columns, rows), where:
          1) columns --- the names of topics in topic model
          2) rows --- the tokens of topic model
          3) data --- content of Phi matrix
        """
        if not self._initialized:
            raise RuntimeError("Model does not exist yet. Use " +
                               "ArtmModel.initialize()/ArtmModel.fit_*()")

        topic_model = self._master.GetTopicModel(model=self._model,
                                                 class_ids=class_ids,
                                                 topic_names=topic_names)
        tokens = [token for token in topic_model[0].token]
        topic_names = [topic_name for topic_name in topic_model[0].topic_name]
        retval = DataFrame(data=topic_model[1],
                           columns=topic_names,
                           index=tokens)

        return retval

    def get_theta(self, remove_theta=False):
        """ArtmModel.get_theta() --- get Theta matrix for training set
        of documents

        Args:
          remove_theta (bool): flag indicates save or remove Theta from model
          after extraction, default=False

        Returns:
          pandas.DataFrame: (data, columns, rows), where:
          1) columns --- the ids of documents, for which the Theta
          matrix was requested
          2) rows --- the names of topics in topic model, that was
          used to create Theta
          3) data --- content of Theta matrix
        """
        if self.cache_theta is False:
            raise ValueError("ArtmModel.get_theta(): cache_theta == False" +
                             "Set ArtmModel.cache_theta = True")
        if not self._initialized:
            raise RuntimeError("ArtmModel.get_theta(): Model does not exist yet. Use " +
                               "ArtmModel.initialize()/ArtmModel.fit_*()")

        theta_matrix = self._master.GetThetaMatrix(self._model, clean_cache=remove_theta)
        document_ids = [item_id for item_id in theta_matrix[0].item_id]
        topic_names = [topic_name for topic_name in theta_matrix[0].topic_name]
        retval = DataFrame(data=theta_matrix[1].transpose(),
                           columns=document_ids,
                           index=topic_names)

        return retval

    def find_theta(self, batches=None, collection_name=None,
                   data_path='', data_format='batches'):
        """ArtmModel.find_theta() --- find Theta matrix for new documents

        Args:
          collection_name (str): the name of text collection (required if
          data_format == 'bow_uci'), default=None
          batches (list of str): list of file names of batches to be processed;
          if not None, than data_format should be 'batches'; format '*.batch',
          default=None
          data_path (str):
          1) if data_format == 'batches' =>
          folder containing batches and dictionary;
          2) if data_format == 'bow_uci' =>
          folder containing docword.txt and vocab.txt files;
          3) if data_format == 'vowpal_wabbit' => file in Vowpal Wabbit format;
          4) if data_format == 'plain_text' => file with text;
          default=''
          data_format (str): the type of input data;
          1) 'batches' --- the data in format of BigARTM;
          2) 'bow_uci' --- Bag-Of-Words in UCI format;
          3) 'vowpal_wabbit' --- Vowpal Wabbit format;
          4) 'plain_text' --- source text;
          default='batches'

        Returns:
          pandas.DataFrame: (data, columns, rows), where:
          1) columns --- the ids of documents, for which the Theta
          matrix was requested
          2) rows --- the names of topics in topic model, that was
          used to create Theta
          3) data --- content of Theta matrix.
        """
        if collection_name is None and data_format == 'bow_uci':
            raise IOError('ArtmModel.find_theta(): No collection name was given')

        if not data_format == 'batches' and batches is not None:
            raise IOError("ArtmModel.find_theta(): batches != None require" +
                          "data_format == batches")

        if not self._initialized:
            raise RuntimeError("ArtmModel.find_theta(): Model does not exist yet. Use " +
                               "ArtmModel.initialize()/ArtmModel.fit_*()")

        target_folder = data_path + '/batches_temp_' + str(random.uniform(0, 1))
        batches_list = []
        if data_format == 'batches':
            if batches is None:
                batches_list = glob.glob(data_path + '/*.batch')
                if len(batches_list) < 1:
                    raise RuntimeError('ArtmModel.find_theta(): No batches were found')
            else:
                batches_list = [data_path + '/' + batch for batch in batches]

        elif data_format == 'bow_uci' or data_format == 'vowpal_wabbit':
            collection_parser_config = create_parser_config(data_path=data_path,
                                                            collection_name=collection_name,
                                                            target_folder=target_folder,
                                                            batch_size=1000,
                                                            data_format=data_format)
            library.Library().ParseCollection(collection_parser_config)
            batches_list = glob.glob(target_folder + '/*.batch')

        elif data_format == 'plain_text':
            raise NotImplementedError()
        else:
            raise IOError('ArtmModel.find_theta(): Unknown data format')

        results = self._master.ProcessBatches(
            pwt=self._model,
            batches=batches_list,
            target_nwt='nwt_hat',
            inner_iterations_count=self._num_document_passes,
            class_ids=self._class_ids,
            theta_matrix_type=library.ProcessBatchesArgs_ThetaMatrixType_External)

        document_ids = [item_id for item_id in results[0].theta_matrix.item_id]
        topic_names = [topic_name for topic_name in results[0].theta_matrix.topic_name]
        retval = DataFrame(data=results[1].transpose(),
                           columns=document_ids,
                           index=topic_names)

        # Remove temp batches folder if necessary
        if not data_format == 'batches':
            shutil.rmtree(target_folder)

        return retval

    def initialize(self, data_path=None, dictionary_name=None):
        """ArtmModel.initialize() --- initialize topic model before learning

        Args:
          data_path (str): name of directory containing BigARTM batches, default=None
          dictionary_name (str): the name of loaded BigARTM collection
          dictionary, default=None

        Note:
          Priority of initialization:
          1) batches in 'data_path'
          2) dictionary
        """
        if data_path is not None:
            self._master.InitializeModel(model_name=self._model,
                                         batch_folder=data_path,
                                         topics_count=self._num_topics,
                                         topic_names=self._topic_names)
        else:
            self._master.InitializeModel(model_name=self._model,
                                         dictionary_name=dictionary_name,
                                         topics_count=self._num_topics,
                                         topic_names=self._topic_names)

        topic_model = self._master.GetTopicModel(model=self._model, use_matrix=False)
        self._topic_names = [topic_name for topic_name in topic_model.topic_name]
        self._initialized = True

        # Remove all info about previous iterations
        self._scores_info = {}
        self._synchronizations_processed = 0
