# This module contains ArtmModel class and helper classes and provides high-level
# Python API for BigARTM Library.
#
# Written for Python 2.7
# Code satisfies the pep8 Python code style guide.
#
# Each change of this file should be tested with pep8 Python style guide checker
# (https://pypi.python.org/pypi/pep8) using command
# > pep8 --first --max-line-length=99 artm_model.py

import collections
from collections import OrderedDict, namedtuple
import csv
import itertools
import json
import glob
import numpy as np
import math
import pandas
from pandas import DataFrame
import scipy.spatial.distance as sp_dist
import shutil
import sklearn.decomposition
from sklearn.decomposition import pca
import sys
import os
from os import path
import random
import urllib2
import uuid

import artm.messages_pb2 as messages_pb2
import artm.library as library
import artm.visualization_ldavis as visualization


###################################################################################################
THETA_REGULARIZER_TYPE = 0
PHI_REGULARIZER_TYPE = 1
GLOB_EPS = 1e-37


def reconfigure_score_in_master(master, score_config, name):
    master_config = messages_pb2.MasterComponentConfig()
    master_config.CopyFrom(master.config())
    for i in range(len(master_config.score_config)):
        if master_config.score_config[i].name == name:
            master_config.score_config[i].config = score_config.SerializeToString()
            break
    master.Reconfigure(master_config)


def create_parser_config(data_path, collection_name, target_folder,
                         batch_size, data_format, dictionary_name='dictionary'):
    collection_parser_config = messages_pb2.CollectionParserConfig()
    collection_parser_config.num_items_per_batch = batch_size
    if data_format == 'bow_uci':
        collection_parser_config.docword_file_path = data_path + 'docword.' + \
          collection_name + '.txt'
        collection_parser_config.vocab_file_path = data_path + 'vocab.' + collection_name + '.txt'
        collection_parser_config.format = library.CollectionParserConfig_Format_BagOfWordsUci
    elif data_format == 'vowpal_wabbit':
        collection_parser_config.docword_file_path = data_path
        collection_parser_config.format = library.CollectionParserConfig_Format_VowpalWabbit
    collection_parser_config.target_folder = target_folder
    collection_parser_config.dictionary_file_name = dictionary_name

    return collection_parser_config


def download_ldavis():
    adress = 'https://raw.githubusercontent.com/romovpa/' + \
        'bigartm/notebook-ideas/notebooks/ldavis/ldavis.js'
    ldavis_js = urllib2.urlopen(address).read()
    with open('../artm/_js/ldavis.js', 'w') as fout:
        fout.write(ldavis_js)


def sym_kl_dist(u, v):
    s = [(x + y) * 0.5 + GLOB_EPS for x, y in zip(u, v)]
    temp1 = 0.5 * sum(a * (math.log(b) if b > GLOB_EPS else 0)
                      for a, b in zip(u, [x / y for x, y in zip(u, s)]))

    temp2 = 0.5 * sum(a * (math.log(b) if b > GLOB_EPS else 0)
                      for a, b in zip(v, [x / y for x, y in zip(v, s)]))

    return temp1 + temp2


###################################################################################################
class Regularizers(object):
    """ Regularizers represents a storage of regularizers in ArtmModel
    (private class).

    Parameters:
    ----------
    - master --- reference to master component object, no default
    """
    def __init__(self, master):
        self._data = {}
        self._master = master

    def add(self, config):
        """ Regularizers.add() --- add regularizer into ArtmModel.
        Parameters:
        ---------
        - config --- an object of ***Regularizer class, no default
        """
        if config.name in self._data:
            print 'Regularizer with name ' + str(config.name) + ' is already exist'
        else:
            regularizer = self._master.CreateRegularizer(config.name, config.type, config.config)
            config.regularizer = regularizer
            self._data[config.name] = config

    def __getitem__(self, name):
        """ Regularizers.__getitem__() --- get regularizer with given name.
        Parameters:
        ---------
        - name --- name of the regularizer.
          Is string, no default
        """
        if name in self._data:
            return self._data[name]
        else:
            print 'No regularizer with name ' + str(config.name)

    @property
    def data(self): return self._data


###################################################################################################
class Scores(object):
    """ Scores represents a storage of scores in ArtmModel (private class).

    Parameters:
    ----------
    - master --- reference to master component object, no default
    """
    def __init__(self, master, model):
        self._data = {}
        self._master = master
        self._model = model

    def add(self, config):
        """ Scores.add() --- add score into ArtmModel.
        Parameters:
        ---------
        - config --- an object of ***Scores class, no default
        """
        if config.name in self._data:
            print 'Score with name ' + str(config.name) + ' is already exist'
        else:
            score = self._master.CreateScore(config.name, config.type, config.config)
            config.model = self._model
            config.score = score
            config.master = self._master
            self._data[config.name] = config

    def __getitem__(self, name):
        """ Scores.__getitem__() --- get score with given name.
        Parameters:
        ----------
        - name --- name of the score.
          Is string, no default
        """
        if name in self._data:
            return self._data[name]
        else:
            print 'No score with name ' + str(config.name)

    @property
    def data(self): return self._data


###################################################################################################
# SECTION OF REGULARIZER CLASSES
###################################################################################################
class SmoothSparsePhiRegularizer(object):
    """ SmoothSparsePhiRegularizer is a regularizer in ArtmModel
    (public class).

    Parameters:
    ----------
    - name --- the identifier of regularizer.
      Is string, default = None

    - tau --- the coefficient of regularization for this regularizer.
      Is double, default = 1.0

    - class_ids --- list of class_ids to regularize. Is list of strings.
      Is default = None

    - topic_names --- list of names of topics to regularize.
      Is list of strings, default = None

    - dictionary_name --- BigARTM collection dictionary.
      Is string, default = None
    """
    def __init__(self, name=None, tau=1.0, class_ids=None,
                 topic_names=None, dictionary_name=None):
        config = messages_pb2.SmoothSparsePhiConfig()
        self._class_ids = []
        self._topic_names = []
        self._dictionary_name = ''

        if name is None:
            name = "SmoothSparsePhiRegularizer:" + uuid.uuid1().urn
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
        self._tau = tau
        self._type = PHI_REGULARIZER_TYPE
        self._config = config
        self._type = library.RegularizerConfig_Type_SmoothSparsePhi
        self._regularizer = None  # Reserve place for the regularizer

    @property
    def name(self):
        return self._name

    @property
    def tau(self):
        return self._tau

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

    @regularizer.setter
    def regularizer(self, regularizer):
        self._regularizer = regularizer

    @tau.setter
    def tau(self, tau):
        self._tau = tau

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
    """ SmoothSparseThetaRegularizer is a regularizer in ArtmModel (public class).
    Parameters:
    ---------
    - name --- the identifier of regularizer.
      Is string, default = None

    - tau --- the coefficient of regularization for this regularizer.
      Is double, default = 1.0

    - topic_names --- list of names of topics to regularize.
      Is list of strings, default = None

    - alpha_iter --- list of additional coefficients of regularization
      on each iteration over document. Should have length equal to
      model.num_document_passes.
      Is list of double, default = None
    """
    def __init__(self, name=None, tau=1.0, topic_names=None, alpha_iter=None):
        config = messages_pb2.SmoothSparseThetaConfig()
        self._topic_names = []
        self._alpha_iter = []

        if name is None:
            name = "SmoothSparseThetaRegularizer:" + uuid.uuid1().urn
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
        self._tau = tau
        self._type = THETA_REGULARIZER_TYPE
        self._config = config
        self._type = library.RegularizerConfig_Type_SmoothSparseTheta
        self._regularizer = None  # Reserve place for the regularizer

    @property
    def name(self):
        return self._name

    @property
    def tau(self):
        return self._tau

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

    @regularizer.setter
    def regularizer(self, regularizer):
        self._regularizer = regularizer

    @tau.setter
    def tau(self, tau):
        self._tau = tau

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
    """ DecorrelatorPhiRegularizer is a regularizer in ArtmModel
    (public class).

    Parameters:
    ----------
    - name --- the identifier of regularizer.
      Is string, default = None

    - tau --- the coefficient of regularization for this regularizer.
      Is double, default = 1.0

    - class_ids --- list of class_ids to regularize.
      Is list of strings, default = None

    - topic_names --- list of names of topics to regularize.
      Is list of strings, default = None
    """
    def __init__(self, name=None, tau=1.0, class_ids=None, topic_names=None):
        config = messages_pb2.DecorrelatorPhiConfig()
        self._class_ids = []
        self._topic_names = []

        if name is None:
            name = "DecorrelatorPhiRegularizer:" + uuid.uuid1().urn
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
        self._tau = tau
        self._type = PHI_REGULARIZER_TYPE
        self._config = config
        self._type = library.RegularizerConfig_Type_DecorrelatorPhi
        self._regularizer = None  # Reserve place for the regularizer

    @property
    def name(self):
        return self._name

    @property
    def tau(self):
        return self._tau

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

    @regularizer.setter
    def regularizer(self, regularizer):
        self._regularizer = regularizer

    @tau.setter
    def tau(self, tau):
        self._tau = tau

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
class LableRegularizationPhiRegularizer(object):
    """ LableRegularizationPhiRegularizer is a regularizer in ArtmModel
    (public class).
    Parameters:
    ----------
    - name --- the identifier of regularizer.
      Is string, default = None

    - tau --- the coefficient of regularization for this regularizer.
      Is double, default = 1.0

    - class_ids --- list of class_ids to regularize.
      Is list of strings, default = None

    - topic_names --- list of names of topics to regularize.
      Is list of strings, default = None

    - dictionary_name --- BigARTM collection dictionary.
      Is string, default = None
    """
    def __init__(self, name=None, tau=1.0, class_ids=None,
                 topic_names=None, dictionary_name=None):
        config = messages_pb2.LableRegularizationPhiConfig()
        self._class_ids = []
        self._topic_names = []
        self._dictionary_name = ''

        if name is None:
            name = "LableRegularizationPhiRegularizer:" + uuid.uuid1().urn
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
        self._tau = tau
        self._type = PHI_REGULARIZER_TYPE
        self._config = config
        self._type = library.RegularizerConfig_Type_LableRegularizationPhi
        self._regularizer = None  # Reserve place for the regularizer

    @property
    def name(self):
        return self._name

    @property
    def tau(self):
        return self._tau

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

    @regularizer.setter
    def regularizer(self, regularizer):
        self._regularizer = regularizer

    @tau.setter
    def tau(self, tau):
        self._tau = tau

    @class_ids.setter
    def class_ids(self, class_ids):
        self._class_ids = class_ids
        config = messages_pb2.LableRegularizationPhiConfig()
        config.CopyFrom(self._config)
        config.ClearField('class_id')
        for class_id in class_ids:
            config.class_id.append(class_id)
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @topic_names.setter
    def topic_names(self, topic_names):
        self._topic_names = topic_names
        config = messages_pb2.LableRegularizationPhiConfig()
        config.CopyFrom(self._config)
        config.ClearField('topic_name')
        for topic_name in topic_names:
            config.topic_name.append(topic_name)
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @dictionary_name.setter
    def dictionary_name(self, dictionary_name):
        self._dictionary_name = dictionary_name
        config = messages_pb2.LableRegularizationPhiConfig()
        config.CopyFrom(self._config)
        config.dictionary_name = dictionary_name
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)


###################################################################################################
class SpecifiedSparsePhiRegularizer(object):
    """ SpecifiedSparsePhiRegularizer is a regularizer in ArtmModel
    (public class).

    Parameters:
    ----------
    - name --- the identifier of regularizer.
      Is string, default = None

    - tau --- the coefficient of regularization for this regularizer.
      Is double, default = 1.0

    - class_id --- class_id to regularize.
      Is string, default = None

    - topic_names --- list of names of topics to regularize.
      Is list of strings, default = None

    - max_elements_count --- number of elements to save in row/column.
      Is int, default = None

    - probability_threshold --- if m elements in row/column summarize into
      value >= probability_threshold, m < n => only these elements would
      be saved.
      Is double, in (0,1), default = None

    - sparse_by_columns --- find max elements in column or in row.
      Is bool, default = True
    """
    def __init__(self, name=None, tau=1.0, class_id=None, topic_names=None,
                 max_elements_count=None, probability_threshold=None, sparse_by_columns=True):
        config = messages_pb2.SpecifiedSparsePhiConfig()
        self._class_id = '@default_class'
        self._topic_names = []
        self._max_elements_count = 20
        self._probability_threshold = 0.99
        self._sparse_by_columns = True

        if name is None:
            name = "SpecifiedSparsePhiRegularizer:" + uuid.uuid1().urn
        if class_id is not None:
            config.class_id = class_id
            self._class_id = class_id
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
                self._topic_names.append(topic_name)
        if max_elements_count is not None:
            config.max_elements_count = max_elements_count
            self._max_elements_count = max_elements_count
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
        self._tau = tau
        self._type = PHI_REGULARIZER_TYPE
        self._config = config
        self._type = library.RegularizerConfig_Type_SpecifiedSparsePhi
        self._regularizer = None  # Reserve place for the regularizer

    @property
    def name(self):
        return self._name

    @property
    def tau(self):
        return self._tau

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
    def max_elements_count(self):
        return self._max_elements_count

    @property
    def probability_threshold(self):
        return self._probability_threshold

    @property
    def sparse_by_columns(self):
        return self._sparse_by_columns

    @property
    def regularizer(self):
        return self._regularizer

    @regularizer.setter
    def regularizer(self, regularizer):
        self._regularizer = regularizer

    @tau.setter
    def tau(self, tau):
        self._tau = tau

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

    @max_elements_count.setter
    def max_elements_count(self, max_elements_count):
        self._max_elements_count = max_elements_count
        config = messages_pb2.SpecifiedSparseRegularizationPhiConfig()
        config.CopyFrom(self._config)
        config.max_elements_count = max_elements_count
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
    """ ImproveCoherencePhiRegularizer is a regularizer in ArtmModel
    (public class).

    Parameters:
    ----------
    - name --- the identifier of regularizer.
      Is string, default = None

    - tau --- the coefficient of regularization for this regularizer.
      Is double, default = 1.0

    - class_ids --- list of class_ids to regularize.
      Is list of strings, default = None

    - topic_names --- list of names of topics to regularize.
      Is list of strings, default = None

    - dictionary_name --- BigARTM collection dictionary.
      Is string, default = None
    """
    def __init__(self, name=None, tau=1.0, class_ids=None,
                 topic_names=None, dictionary_name=None):
        config = messages_pb2.ImproveCoherencePhiConfig()
        self._class_ids = []
        self._topic_names = []
        self._dictionary_name = ''

        if name is None:
            name = "ImproveCoherencePhiRegularizer:" + uuid.uuid1().urn
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
        self._tau = tau
        self._type = PHI_REGULARIZER_TYPE
        self._config = config
        self._type = library.RegularizerConfig_Type_ImproveCoherencePhi
        self._regularizer = None  # Reserve place for the regularizer

    @property
    def name(self):
        return self._name

    @property
    def tau(self):
        return self._tau

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

    @regularizer.setter
    def regularizer(self, regularizer):
        self._regularizer = regularizer

    @tau.setter
    def tau(self, tau):
        self._tau = tau

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
    """ SparsityPhiScore is a score in ArtmModel (public class).

    Parameters:
    ----------
    - name --- the identifier of score.
      Is string, default = None

    - class_id --- class_id to score.
       Is string, default = None

    - topic_names --- list of names of topics to score.
      Is list of strings, default = None

    - eps --- the tolerance const, everything < eps considered to be zero.
      Is double, default = None
    """
    def __init__(self, name=None, class_id=None, topic_names=None, eps=None):
        config = messages_pb2.SparsityPhiScoreConfig()
        self._class_id = '@default_class'
        self._topic_names = []
        self._eps = GLOB_EPS

        if name is None:
            name = "SparsityPhiScore:" + uuid.uuid1().urn
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

    @model.setter
    def model(self, model):
        self._model = model

    @score.setter
    def score(self, score):
        self._score = score

    @master.setter
    def master(self, master):
        self._master = master

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
    """ SparsityThetaScore is a score in ArtmModel (public class).

    Parameters:
    - name --- the identifier of score.
      Is string, default = None

    - topic_names --- list of names of topics to score.
      Is list of strings, default = None

    - eps --- the tolerance const, everything < eps considered to be zero.
      Is double, default = None
    """
    def __init__(self, name=None, topic_names=None, eps=None):
        config = messages_pb2.SparsityThetaScoreConfig()
        self._topic_names = []
        self._eps = GLOB_EPS

        if name is None:
            name = "SparsityThetaScore:" + uuid.uuid1().urn
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

    @model.setter
    def model(self, model):
        self._model = model

    @score.setter
    def score(self, score):
        self._score = score

    @master.setter
    def master(self, master):
        self._master = master

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
    """ PerplexityScore is a score in ArtmModel (public class).

    Parameters:
    ----------
    - name --- the identifier of score.
      Is string, default = None

    - class_id --- class_id to score.
      Is string, default = None

    - topic_names --- list of names of topics to score Theta sparsity.
      Is list of strings, default = None

    - eps --- the tolerance const for Theta sparsity, everything < eps
      considered to be zero.
      Is double, default = None

    - dictionary_name --- BigARTM collection dictionary.
      Is string, default = None

    - use_unigram_document_model --- use uni-gram document/collection model
      if token's counter == 0.
      Is bool, default = None
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
            name = "PerplexityScore:" + uuid.uuid1().urn
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

    @model.setter
    def model(self, model):
        self._model = model

    @score.setter
    def score(self, score):
        self._score = score

    @master.setter
    def master(self, master):
        self._master = master

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
    """ ItemsProcessedScore is a score in ArtmModel (public class).

    Parameters:
    ----------
    - name --- the identifier of score.
      Is string, default = None
    """
    def __init__(self, name=None):
        config = messages_pb2.ItemsProcessedScoreConfig()

        if name is None:
            name = "PerplexityScore:" + uuid.uuid1().urn

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

    @model.setter
    def model(self, model):
        self._model = model

    @score.setter
    def score(self, score):
        self._score = score

    @master.setter
    def master(self, master):
        self._master = master


###################################################################################################
class TopTokensScore(object):
    """ TopTokensScore is a score in ArtmModel (public class).

    Parameters:
    ----------
    - name --- the identifier of score.
      Is string, default = None

    - class_id --- class_id to score.
      Is string, default = None

    - topic_names --- list of names of topics to score Theta sparsity.
      Is list of strings, default = None

    - num_tokens --- Number of tokens with max probability in each topic.
      Is int, default = None

    - dictionary_name --- BigARTM collection dictionary.
      Is string, default = None
    """
    def __init__(self, name=None, class_id=None, topic_names=None,
                 num_tokens=None, dictionary_name=None):
        config = messages_pb2.TopTokensScoreConfig()
        self._class_id = '@default_class'
        self._topic_names = []
        self._num_tokens = 10
        self._dictionary_name = ''

        if name is None:
            name = "TopTokensScore:" + uuid.uuid1().urn
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

    @model.setter
    def model(self, model):
        self._model = model

    @score.setter
    def score(self, score):
        self._score = score

    @master.setter
    def master(self, master):
        self._master = master

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
    """ ThetaSnippetScore is a score in ArtmModel (public class).

    Parameters:
    ----------
    - name --- the identifier of score.
      Is string, default = None

    - item_ids --- list of names of items to show.
      Is list of ints, default = None

    - num_items --- number of theta vectors to show from the
      beginning (no sense if item_ids given).
      Is int, default = None
    """
    def __init__(self, name=None, item_ids=None, num_items=None):
        config = messages_pb2.ThetaSnippetScoreConfig()
        self._item_ids = []
        self._num_items = 10

        if name is None:
            name = "ThetaSnippetScore:" + uuid.uuid1().urn
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

    @model.setter
    def model(self, model):
        self._model = model

    @score.setter
    def score(self, score):
        self._score = score

    @master.setter
    def master(self, master):
        self._master = master

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
    """ TopicKernelScore is a score in ArtmModel (public class).

    Parameters:
    - name --- the identifier of score.
      Is string, default = None

    - class_id --- class_id to score.
      Is string, default = None

    - topic_names --- list of names of topics to score Theta sparsity.
      Is list of strings, default = None

    - eps --- the tolerance const for counting, everything < eps
      considered to be zero.
      Is double, default = None

    - dictionary_name --- BigARTM collection dictionary_name.
      Is string, default = None
    - probability_mass_threshold --- the threshold for p(t|w) values to get
      token into topic kernel.
      Is double, in (0,1), default = None
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
            name = "TopicKernelScore:" + uuid.uuid1().urn
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

    @model.setter
    def model(self, model):
        self._model = model

    @score.setter
    def score(self, score):
        self._score = score

    @master.setter
    def master(self, master):
        self._master = master

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
        score_config.dictionary = dictionary_name
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
    """ SparsityPhiScoreInfo represents a result of counting SparsityPhiScore
    (private class).

    Parameters:
    ----------
    - score --- reference to score object, no default
    """
    def __init__(self, score):
        self._name = score.name
        self._value = []
        self._zero_tokens = []
        self._total_tokens = []

    def add(self, score=None):
        """ SparsityPhiScoreInfo.add() --- add info about score after
        synchronization.

        Parameters:
        ----------
        - score --- reference to score object,
          default = None (means "Add None values")
        """
        if score is not None:
            _data = messages_pb2.SparsityPhiScore()
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
        """ Returns value of Phi sparsity on synchronizations.
        Is list of scalars
        """
        return self._value

    @property
    def zero_tokens(self):
        """ Returns number of zero rows in Phi on synchronizations.
        Is list of scalars
        """
        return self._zero_tokens

    @property
    def total_tokens(self):
        """ Returns total number of rows in Phi on synchronizations.
        Is list of scalars
        """
        return self._total_tokens


###################################################################################################
class SparsityThetaScoreInfo(object):
    """ SparsityThetaScoreInfo represents a result of counting
    SparsityThetaScore (private class).

    Parameters:
    ----------
    - score --- reference to score object, no default
    """
    def __init__(self, score):
        self._name = score.name
        self._value = []
        self._zero_topics = []
        self._total_topics = []

    def add(self, score=None):
        """ SparsityThetaScoreInfo.add() --- add info about score
        after synchronization.

        Parameters:
        ----------
        - score --- reference to score object,
          default = None (means "Add None values")
        """
        if score is not None:
            _data = messages_pb2.SparsityThetaScore()
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
        """ Returns value of Theta sparsity on synchronizations.
        Is list of scalars
        """
        return self._value

    @property
    def zero_topics(self):
        """ Returns number of zero rows in Theta on synchronizations.
        Is list of scalars
        """
        return self._zero_topics

    @property
    def total_topics(self):
        """ Returns total number of rows in Theta on synchronizations.
        Is list of scalars
        """
        return self._total_topics


###################################################################################################
class PerplexityScoreInfo(object):
    """ PerplexityScoreInfo represents a result of counting PerplexityScore
    (private class).

    Parameters:
    ----------
    - score --- reference to score object, no default
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
        """ PerplexityScoreInfo.add() --- add info about score after
        synchronization.

        Parameters:
        ----------
        - score --- reference to score object,
          default = None (means "Add None values")
        """
        if score is not None:
            _data = messages_pb2.PerplexityScore()
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
        """ Returns value of perplexity on synchronizations.
        Is list of scalars
        """
        return self._value

    @property
    def raw(self):
        """ Returns raw value in formula of perplexity on synchronizations.
        Is list of scalars
        """
        return self._raw

    @property
    def normalizer(self):
        """ normalizer value in formula of perplexity on synchronizations.
        Is list of scalars
        """
        return self._normalizer

    @property
    def zero_tokens(self):
        """ number of tokens with zero counters on synchronizations.
        Is list of scalars
        """
        return self._zero_tokens

    @property
    def theta_sparsity_value(self):
        """ Returns Theta sparsity value on synchronizations.
        Is list of scalars
        """
        return self._theta_sparsity_value

    @property
    def theta_sparsity_zero_topics(self):
        """ Returns number of zero rows in Theta on synchronizations.
        Is list of scalars
        """
        return self._theta_sparsity_zero_topics

    @property
    def theta_sparsity_total_topics(self):
        """ Returns total number of rows in Theta on synchronizations.
        Is list of scalars
        """
        return self._theta_sparsity_total_topics


###################################################################################################
class ItemsProcessedScoreInfo(object):
    """ ItemsProcessedScoreInfo represents a result of counting
    ItemsProcessedScore (private class).

    Parameters:
    ----------
    - score --- reference to score object, no default
    """
    def __init__(self, score):
        self._name = score.name
        self._value = []

    def add(self, score=None):
        """ ItemsProcessedScoreInfo.add() --- add info about score
        after synchronization.

        Parameters:
        ----------
        - score --- reference to score object,
          default = None (means "Add None values")
        """
        if score is not None:
            _data = messages_pb2.ItemsProcessedScore()
            _data = score.score.GetValue(score._model)
            self._value.append(_data.value)
        else:
            self._value.append(None)

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        """ Returns total number of processed documents on synchronizations.
        Is list of scalars
        """
        return self._value


###################################################################################################
class TopTokensScoreInfo(object):
    """ TopTokensScoreInfo represents a result of counting TopTokensScore
    (private class).

    Parameters:
    ----------
    - score --- reference to score object, no default
    """
    def __init__(self, score):
        self._name = score.name
        self._num_tokens = []
        self._topic_info = []
        self._average_coherence = []

    def add(self, score=None):
        """ TopTokensScoreInfo.add() --- add info about score
        after synchronization.

        Parameters:
        ----------
        - score --- reference to score object,
          default = None (means "Add None values")
        """
        if score is not None:
            _data = messages_pb2.TopTokensScore()
            _data = score.score.GetValue(score._model)

            self._num_tokens.append(_data.num_entries)

            self._topic_info.append({})
            index = len(self._topic_info) - 1
            topic_index = -1
            for topic_name in list(OrderedDict.fromkeys(_data.topic_name)):
                topic_index += 1
                tokens = []
                weights = []
                for i in range(_data.num_entries):
                    if _data.topic_name[i] == topic_name:
                        tokens.append(_data.token[i])
                        weights.append(_data.weight[i])
                coherence = -1
                if len(_data.coherence.value) > 0:
                    coherence = _data.coherence.value[topic_index]
                self._topic_info[index][topic_name] = \
                    namedtuple('TopTokensScoreTuple', ['tokens', 'weights', 'coherence'])
                self._topic_info[index][topic_name].tokens = tokens
                self._topic_info[index][topic_name].weights = weights
                self._topic_info[index][topic_name].coherence = coherence

            self._average_coherence. append(_data.average_coherence)
        else:
            self._num_tokens.append(None)
            self._topic_info.append(None)
            self._average_coherence.append(None)

    @property
    def name(self):
        return self._name

    @property
    def num_tokens(self):
        """ Returns reqested number of top tokens in each topic on
        synchronizations.
        Is list of scalars
        """
        return self._num_tokens

    @property
    def topic_info(self):
        """ Returns information about top tokens per topic on synchronizations.
        Is list of sets. Set contains information about topics,
        key --- name of topic, value --- named tuple:

        - *.topic_info[sync_index][topic_name].tokens --- list of top tokens
          for this topic.

        - *.topic_info[sync_index][topic_name].weights --- list of weights
          (probabilities), corresponds the tokens.
        - *.topic_info[sync_index][topic_name].coherence --- the coherency
          of topic due to it's top tokens.
        """
        return self._topic_info

    @property
    def average_coherence(self):
        """ Returns average coherence of top tokens in all requested topics
        on synchronizations.
        Is list of scalars
        """
        return self._average_coherence


###################################################################################################
class TopicKernelScoreInfo(object):
    """ TopicKernelScoreInfo represents a result of counting TopicKernelScore
    (private class).

    Parameters:
    ----------
    - score --- reference to score object, no default
    """
    def __init__(self, score):
        self._name = score.name
        self._topic_info = []
        self._average_coherence = []
        self._average_size = []
        self._average_contrast = []
        self._average_purity = []

    def add(self, score=None):
        """ TopicKernelScoreInfo.add() --- add info about score after
        synchronization.

        Parameters:
        ----------
        - score --- reference to score object,
          default = None (means "Add None values")
        """
        if score is not None:
            _data = messages_pb2.TopicKernelScore()
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
                    namedtuple('TopicKernelScoreTuple',
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
        """ Returns information about kernel tokens per topic on
        synchronizations. Is list of sets. Set contains information
        about topics, key --- name of topic, value --- named tuple:

        - *.topic_info[sync_index][topic_name].tokens --- list of
          kernel tokens for this topic.
        - *.topic_info[sync_index][topic_name].size --- size of
          kernel for this topic.
        - *.topic_info[sync_index][topic_name].contrast --- contrast of
          kernel for this topic.
        - *.topic_info[sync_index][topic_name].purity --- purity of kernel
          for this topic.
        - *.topic_info[sync_index][topic_name].coherence --- the coherency of
          topic due to it's kernel.
        """
        return self._topic_info

    @property
    def average_coherence(self):
        """ Returns average coherence of kernel tokens in all requested
        topics on synchronizations.
        Is list of scalars
        """
        return self._average_coherence

    @property
    def average_size(self):
        """ Returns average kernel size of all requested topics on
        synchronizations.
        Is list of scalars
        """
        return self._average_size

    @property
    def average_contrast(self):
        """ Returns average kernel contrast of all requested topics on
        synchronizations.
        Is list of scalars
        """
        return self._average_contrast

    @property
    def average_purity(self):
        """ Returns average kernel purity of all requested topics on
        synchronizations.
        Is list of scalars
        """
        return self._average_purity


###################################################################################################
class ThetaSnippetScoreInfo(object):
    """ ThetaSnippetScoreInfo represents a result of counting
    ThetaSnippetScore (private class).

    Parameters:
    ----------
    - score --- reference to score object, no default
    """
    def __init__(self, score):
        self._name = score.name
        self._document_ids = []
        self._snippet = []

    def add(self, score=None):
        """ ThetaSnippetScoreInfo.add() --- add info about score after
        synchronization.

        Parameters:
        - score --- reference to score object,
          default = None (means "Add None values")
        """
        if score is not None:
            _data = messages_pb2.ThetaSnippetScore()
            _data = score.score.GetValue(score._model)

            self._document_ids .append([item_id for item_id in _data.item_id])
            self._snippet.append(
                [[theta_td for theta_td in theta_d.value] for theta_d in _data.values])
        else:
            self._document_ids.append(None)
            self._snippet.append(None)

    @property
    def name(self): return self._name

    @property
    def snippet(self):
        """ Returns the snippet (part) of Theta corresponds to documents from
        document_ids.
        Is list of lists of scalars, each internal list --- theta_d vector
        for document d, in direct order of document_ids
        """
        return self._snippet

    @property
    def document_ids(self):
        """ Returns ids of documents in snippet on synchronizations.
        Is list of scalars
        """
        return self._document_ids


###################################################################################################
# SECTION OF ARTM MODEL CLASS
###################################################################################################
class ArtmModel(object):
    """ ArtmModel represents a topic model (public class).
    Parameters:
    -----------
    - num_processors --- how many threads will be used for model training.
    Is int, default = 0 (means that number of threads will be
    detected by the library)

    - topic_names --- names of topics in model.
    Is list of strings, default = []

    - topics_count --- number of topics in model (is used if
    topic_names == []). Is int, default = 10

    - class_ids --- list of class_ids and their weights to be used in model.
    Is dict, key --- class_id, value --- weight, default = {}

    - num_document_passes --- number of iterations over each document
    during processing/ Is int, default = 1

    - cache_theta --- save or not the Theta matrix in model. Necessary
    if ArtmModel.get_theta() usage expects. Is bool, default = True

    Important public fields:
    ----------
    - regularizers --- contains dict of regularizers, included into model
    - scores --- contains dict of scores, included into model
    - scores_info --- contains dict of scoring results;
    key --- score name, value --- ScoreInfo object, which contains info about
    values of score on each synchronization in list

    NOTE:
    ----------
    - Here and anywhere in BigARTM empty topic_names or class_ids means that
      model (or regularizer, or score) should use all topics or class_ids.
    - If some fields of regularizers or scores are not defined by
      user --- internal library defaults would be used.
    - If field 'topics_name' == [], it will be generated by BigARTM and will
      be available using ArtmModel.topics_name().
    """

# ========== CONSTRUCTOR ==========
    def __init__(self, num_processors=0, topic_names=[], topics_count=10,
                 class_ids={}, num_document_passes=1, cache_theta=True):
        self._num_processors = 0
        self._topics_count = 10
        self._topic_names = []
        self._class_ids = {}
        self._num_document_passes = 1
        self._cache_theta = True

        if num_processors > 0:
            self._num_processors = num_processors

        if topics_count > 0:
            self._topics_count = topics_count

        if len(class_ids) > 0:
            self._class_ids = class_ids

        if num_document_passes > 0:
            self._num_document_passes = num_document_passes

        if isinstance(cache_theta, bool):
            self._cache_theta = cache_theta

        if len(topic_names) > 0:
            self._topic_names = topic_names
            self._topics_count = len(topic_names)

        self._master = library.MasterComponent()
        self._master.config().processors_count = self._num_processors
        self._master.config().cache_theta = cache_theta
        self._master.Reconfigure()

        self._model = 'pwt'
        self._regularizers = Regularizers(self._master)
        self._scores = Scores(self._master, self._model)

        self._scores_info = {}
        self._synchronizations_processed = 0
        self._was_initialized = False

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
    def tokens_count(self):
        return self._tokens_count

    @property
    def topics_count(self):
        return self._topics_count

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
            print 'Number of processors should be a positive integer, skip update'
        else:
            self._num_processors = num_processors
            self._master.config().processors_count = num_processors
            self._master.Reconfigure()

    @num_document_passes.setter
    def num_document_passes(self, num_document_passes):
        if num_document_passes <= 0 or not isinstance(num_document_passes, int):
            print 'Number of passes through documents should be a positive integer, skip update'
        else:
            self._num_document_passes = num_document_passes

    @cache_theta.setter
    def cache_theta(self, cache_theta):
        if not isinstance(cache_theta, bool):
            print 'cache_theta should be bool, skip update'
        else:
            self._cache_theta = cache_theta
            self._master.config().cache_theta = cache_theta
            self._master.Reconfigure()

    @topics_count.setter
    def topics_count(self, topics_count):
        if topics_count <= 0 or not isinstance(topics_count, int):
            print 'Number of topics should be a positive integer, skip update'
        else:
            self._topics_count = topics_count

    @topic_names.setter
    def topic_names(self, topic_names):
        if len(topic_names) < 0:
            print 'Number of topic names should be non-negative, skip update'
        else:
            self._topic_names = topic_names
            self._topics_count = len(topic_names)

    @class_ids.setter
    def class_ids(self, class_ids):
        if len(class_ids) < 0:
            print 'Number of (class_id, class_weight) pairs shoul be non-negative, skip update'
        else:
            self._class_ids = class_ids

# ========== METHODS ==========
    def parse(self, collection_name=None, data_path='', data_format='batches',
              batch_size=1000, dictionary_name='dictionary'):
        """ ArtmModel.fit() --- proceed the learning of topic model

        Parameters:
        ----------
        - collection_name --- the name of text collection
          (required if data_format == 'bow_uci').
          Is string, default = None

        - data_path --- 1) if data_format == 'bow_uci' =>
          folder containing 'docword.collection_name.txt'
          and vocab.collection_name.txt files
          2) if data_format == 'vowpal_wabbit' => file in Vowpal Wabbit format
          3) if data_format == 'plain_text' => file with text
          Is string, default = ''

        - data_format --- the type of input data:
          1) 'bow_uci' --- Bag-Of-Words in UCI format
          2) 'vowpal_wabbit' --- Vowpal Wabbit format
          3) 'plain_text' --- source text
          Is string, default = 'bow_uci'

        - batch_size --- number of documents to be stored in each batch.
          Is int, default = 1000

        - dictionary_name --- the name of BigARTM dictionary with information
          about collection, that will be gathered by the library parser.
          Is string, default = 'dictionary'

        #- gather_cooc --- find or not the info about the token pairwise
        #  co-occuracies.
        #  Is bool, default=False

        #- cooc_tokens --- tokens to collect cooc info (has sense if
        #  gather_cooc is True).
        #  Is list of lists, each internal list represents token and contain
        #  two strings --- token and its class_id, default = []

        Note:
        ----------
        Gathering tokens co-ocurracies information is experimental and will be
        changed in the next release.
        """
        if collection_name is None and data_format == 'bow_uci':
                print 'No collection name was given, skip model.fit_offline()'

        if data_format == 'bow_uci' or data_format == 'vowpal_wabbit':
            collection_parser_config = create_parser_config(data_path,
                                                            collection_name,
                                                            collection_name,
                                                            batch_size,
                                                            data_format,
                                                            dictionary_name)
            #collection_parser_config.gather_cooc = gather_cooc
            #for token in cooc_tokens:
            #    collection_parser_config.cooccurrence_token.append(token)
            unique_tokens = library.Library().ParseCollection(collection_parser_config)

        elif data_format == 'plain_text':
            raise NotImplementedError()
        else:
            print 'Unknown data format, skip model.parse()'

    def load_dictionary(self, dictionary_path=None):
        """ ArtmModel.load_dictionary() --- load and return the BigARTM
        dictionary of the collection

        Parameters:
        ----------
        - dictionary_path --- full file name of the dictionary.
          Is string, default = None
        """
        if dictionary_path is not None:
            unique_tokens = library.Library().LoadDictionary(dictionary_path)
            return self._master.CreateDictionary(unique_tokens)
        else:
            print 'dictionary path is None, skip loading dictionary.'

    def fit_offline(self, collection_name=None, batches=None, data_path='',
                    num_collection_passes=1, decay_weight=0.0, apply_weight=1.0,
                    reset_theta_scores=False, data_format='batches', batch_size=1000):
        """ ArtmModel.fit_offline() --- proceed the learning of
        topic model in off-line mode

        Parameters:
        ----------
        - collection_name --- the name of text collection
          (required if data_format == 'bow_uci').
          Is string, default = None

        - batches --- list of file names of batches to be processed.
          If not None, than data_format should be 'batches'.
          Is list of strings in format '*.batch', default = None

        - data_path --- 1) if data_format == 'batches' =>
          folder containing batches and dictionary
          2) if data_format == 'bow_uci' => folder containing
            docword.collection_name.txt and vocab.collection_name.txt files
          3) if data_format == 'vowpal_wabbit' => file in Vowpal Wabbit format
          4) if data_format == 'plain_text' => file with text
          Is string, default = ''

        - num_collection_passes --- number of iterations over whole
          given collection.
          Is int, default = 1

        - decay_weight --- coefficient for applying old n_wt counters.
          Is int, default = 0.0 (apply_weight + decay_weight = 1.0)

        - apply_weight --- coefficient for applying new n_wt counters.
          Is int, default = 1.0 (apply_weight + decay_weight = 1.0)

        - reset_theta_scores --- reset accumulated Theta scores
          before learning.
          Is bool, default = False

        - data_format --- the type of input data:
          1) 'batches' --- the data in format of BigARTM
          2) 'bow_uci' --- Bag-Of-Words in UCI format
          3) 'vowpal_wabbit' --- Vowpal Wabbit format
          4) 'plain_text' --- source text
          Is string, default = 'batches'

        Next argument has sense only if data_format is not 'batches'
        (e.g. parsing is necessary).
        - batch_size --- number of documents to be stored ineach batch.
          Is int, default = 1000

        Note:
        ----------
        ArtmModel.initialize() should be proceed before first call
        ArtmModel.fit_offline(), or it will be initialized by dictionary
        during first call.
        """
        if collection_name is None and data_format == 'bow_uci':
            print 'No collection name was given, skip model.fit_offline()'

        if not data_format == 'batches' and batches is not None:
            print "batches != None require data_format == 'batches'"

        unique_tokens = messages_pb2.DictionaryConfig()
        target_folder = data_path + '/batches_temp_' + str(random.uniform(0, 1))
        batches_list = []
        if data_format == 'batches':
            if batches is None:
                batches_list = glob.glob(data_path + "/*.batch")
                if len(batches_list) < 1:
                    print 'No batches were found, skip model.fit_offline()'
                    return
                print 'fit_offline() found ' + str(len(batches_list)) + ' batches'
                unique_tokens = library.Library().LoadDictionary(data_path + '/dictionary')
            else:
                batches_list = [data_path + '/' + batch for batch in batches]

        elif data_format == 'bow_uci' or data_format == 'vowpal_wabbit':
            collection_parser_config = create_parser_config(data_path,
                                                            collection_name,
                                                            target_folder,
                                                            batch_size,
                                                            data_format)
            unique_tokens = library.Library().ParseCollection(collection_parser_config)
            batches_list = glob.glob(target_folder + "/*.batch")

        elif data_format == 'plain_text':
            raise NotImplementedError()
        else:
            print 'Unknown data format, skip model.fit_offline()'

        if not self._was_initialized:
            self.initialize(dictionary=self._master.CreateDictionary(unique_tokens))

        theta_regularizers, phi_regularizers = {}, {}
        for name, config in self._regularizers.data.iteritems():
            if config.type == THETA_REGULARIZER_TYPE:
                theta_regularizers[name] = config.tau
            else:
                phi_regularizers[name] = config.tau

        for iter in range(num_collection_passes):
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
                    if (self.scores[name].type == library.ScoreConfig_Type_SparsityPhi):
                        self._scores_info[name] = SparsityPhiScoreInfo(self.scores[name])
                    elif (self.scores[name].type == library.ScoreConfig_Type_SparsityTheta):
                        self._scores_info[name] = SparsityThetaScoreInfo(self.scores[name])
                    elif (self.scores[name].type == library.ScoreConfig_Type_Perplexity):
                        self._scores_info[name] = PerplexityScoreInfo(self.scores[name])
                    elif (self.scores[name].type == library.ScoreConfig_Type_ThetaSnippet):
                        self._scores_info[name] = ThetaSnippetScoreInfo(self.scores[name])
                    elif (self.scores[name].type == library.ScoreConfig_Type_ItemsProcessed):
                        self._scores_info[name] = ItemsProcessedScoreInfo(self.scores[name])
                    elif (self.scores[name].type == library.ScoreConfig_Type_TopTokens):
                        self._scores_info[name] = TopTokensScoreInfo(self.scores[name])
                    elif (self.scores[name].type == library.ScoreConfig_Type_TopicKernel):
                        self._scores_info[name] = TopicKernelScoreInfo(self.scores[name])

                    for i in range(self._synchronizations_processed - 1):
                        self._scores_info[name].add()

                self._scores_info[name].add(self.scores[name])

        # Remove temp batches folder if it necessary
        if not data_format == 'batches':
            shutil.rmtree(target_folder)

    def fit_online(self, collection_name=None, batches=None, data_path='',
                   tau0=1024.0, kappa=0.7, update_every=1, reset_theta_scores=False,
                   data_format='batches', batch_size=1000):
        """ ArtmModel.fit_online() --- proceed the learning of topic model
        in on-line mode

        Parameters:
        ----------
        - collection_name --- the name of text collection
          (required if data_format == 'bow_uci').
          Is string, default = None

        - batches --- list of file names of batches to be processed.
          If not None, than data_format should be 'batches'.
          Is list of strings in format '*.batch', default = None

        - data_path --- 1) if data_format == 'batches' =>
          folder containing batches and dictionary
          2) if data_format == 'bow_uci' => folder containing
          docword.collection_name.txt and vocab.collection_name.txt files
          3) if data_format == 'vowpal_wabbit' => file in Vowpal Wabbit format
          4) if data_format == 'plain_text' => file with text
          Is string, default = ''

        - update_every --- the number of batches; model will be updated
          once per it.
          Is int, default = 1

        - tau0 --- coefficient (see kappa).
          Is float, default = 1024.0

        - kappa --- power for tau0.
          Is float, default = 0.7

          The formulas for decay_weight and apply_weight:
          update_count = current_processed_docs / (batch_size * update_every)
          rho = pow(tau0 + update_count, -kappa)
          decay_weight = 1-rho
          apply_weight = rho

        - reset_theta_scores --- reset accumulated Theta scores before
          learning.
          Is bool, default = False

        - data_format --- the type of input data:
          1) 'batches' --- the data in format of BigARTM
          2) 'bow_uci' --- Bag-Of-Words in UCI format
          3) 'vowpal_wabbit' --- Vowpal Wabbit format
          4) 'plain_text' --- source text
          Is string, default = 'batches'

        Next argument has sense only if data_format is not 'batches'
        (e.g. parsing is necessary).
        - batch_size --- number of documents to be stored in each batch.
        Is int, default = 1000

        Note:
        ----------
        ArtmModel.initialize() should be proceed before first call
        ArtmModel.fit_online(), or it will be initialized by dictionary
        during first call.
        """
        if collection_name is None and data_format == 'bow_uci':
                print 'No collection name was given, skip model.fit_online()'

        if not data_format == 'batches' and batches is not None:
            print "batches != None require data_format == 'batches'"

        unique_tokens = messages_pb2.DictionaryConfig()
        target_folder = data_path + '/batches_temp_' + str(random.uniform(0, 1))
        batches_list = []
        if data_format == 'batches':
            if batches is None:
                batches_list = glob.glob(data_path + "/*.batch")
                if len(batches_list) < 1:
                    print 'No batches were found, skip model.fit_online()'
                    return
                print 'fit_online() found ' + str(len(batches_list)) + ' batches'
                unique_tokens = library.Library().LoadDictionary(data_path + '/dictionary')
            else:
                batches_list = [data_path + '/' + batch for batch in batches]

        elif data_format == 'bow_uci' or data_format == 'vowpal_wabbit':
            collection_parser_config = create_parser_config(data_path,
                                                            collection_name,
                                                            target_folder,
                                                            batch_size,
                                                            data_format)
            unique_tokens = library.Library().ParseCollection(collection_parser_config)
            batches = glob.glob(target_folder + "/*.batch")

        elif data_format == 'plain_text':
            raise NotImplementedError()
        else:
            print 'Unknown data format, skip model.fit_online()'

        if not self._was_initialized:
            self.initialize(dictionary=self._master.CreateDictionary(unique_tokens))

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
                decay_weight, apply_weight = 1-rho, rho

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
                        if (self.scores[name].type == library.ScoreConfig_Type_SparsityPhi):
                            self._scores_info[name] = SparsityPhiScoreInfo(self.scores[name])
                        elif (self.scores[name].type == library.ScoreConfig_Type_SparsityTheta):
                            self._scores_info[name] = SparsityThetaScoreInfo(self.scores[name])
                        elif (self.scores[name].type == library.ScoreConfig_Type_Perplexity):
                            self._scores_info[name] = PerplexityScoreInfo(self.scores[name])
                        elif (self.scores[name].type == library.ScoreConfig_Type_ThetaSnippet):
                            self._scores_info[name] = ThetaSnippetScoreInfo(self.scores[name])
                        elif (self.scores[name].type == library.ScoreConfig_Type_ItemsProcessed):
                            self._scores_info[name] = ItemsProcessedScoreInfo(self.scores[name])
                        elif (self.scores[name].type == library.ScoreConfig_Type_TopTokens):
                            self._scores_info[name] = TopTokensScoreInfo(self.scores[name])
                        elif (self.scores[name].type == library.ScoreConfig_Type_TopicKernel):
                            self._scores_info[name] = TopicKernelScoreInfo(self.scores[name])

                        for i in range(self._synchronizations_processed - 1):
                            self._scores_info[name].add()

                    self._scores_info[name].add(self.scores[name])

        # Remove temp batches folder if it necessary
        if not data_format == 'batches':
            shutil.rmtree(target_folder)

    def save(self, file_name='artm_model'):
        """ ArtmModel.save() --- save the topic model to disk.

        Parameters:
        ----------
        - file_name --- the name of file to store model.
          Is string, default = 'artm_model'
        """
        if not self._was_initialized:
            print 'Model does not exist yet. Use ArtmModel.initialize()/ArtmModel.fit_*()'
            return

        if os.path.isfile(file_name):
            os.remove(file_name)
        self._master.ExportModel(self._model, file_name)

    def load(self, file_name):
        """ ArtmModel.load() --- load the topic model,
        saved by ArtmModel.save(), from disk.

        Parameters:
        ----------
        - file_name --- the name of file containing model.
          Is string, no default

        Note:
        ----------
        Loaded model will overwrite ArtmModel.topic_names and
        ArtmModel.topics_count fields. Also it will empty
        ArtmModel.scores_info.
        """
        self._master.ImportModel(self._model, file_name)
        self._was_initialized = True
        args = messages_pb2.GetTopicModelArgs()
        args.request_type = library.GetTopicModelArgs_RequestType_TopicNames
        topic_model = self._master.GetTopicModel(model=self._model, args=args)
        self._topic_names = [topic_name for topic_name in topic_model.topic_name]
        self._topics_count = topic_model.topics_count

        # Remove all info about previous iterations
        self._scores_info = {}
        self._synchronizations_processed = 0

    def to_csv(self, file_name='artm_model.csv'):
        """ ArtmModel.to_csv() --- save the topic model to disk in
        .csv format (can't be loaded back).

        Parameters:
        ----------
        - file_name --- the name of file to store model.
          Is string, default = 'artm_model.csv'
        """
        if not self._was_initialized:
            print 'Model does not exist yet. Use ArtmModel.initialize()/ArtmModel.fit_*()'
            return

        if os.path.isfile(file_name):
            os.remove(file_name)

        with open(file_name, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar=';', quoting=csv.QUOTE_MINIMAL)
            if len(self._topic_names) > 0:
                writer.writerow(['Token'] + ['Class ID'] +
                                ['TOPIC: ' + topic_name for topic_name in self._topic_names])

            topic_model = self._master.GetTopicModel(self._model)
            for index in range(len(topic_model.token_weights)):
                writer.writerow(
                    [topic_model.token[index]] + [topic_model.class_id[index]] +
                    [token_weight for token_weight in topic_model.token_weights[index].value])

    def get_theta(self, remove_theta=False):
        """ ArtmModel.get_theta() --- get Theta matrix for training set
        of documents.

        Parameters:
        ----------
        - remove_theta --- flag indicates save or remove Theta from model
          after extraction.
          Is bool, default = False

        Returns:
        ----------
        - DataFrame (data, columns, rows), where:

          1) columns --- the ids of documents, for which the Theta
          matrix was requested

          2) rows --- the names of topics in topic model, that was
          used to create Theta

          3) data --- content of Theta matrix.
        """
        if self.cache_theta is False:
            print 'ArtmModel.cache_theta == False, skip get_theta().' + \
                  'Set ArtmModel.cache_theta = True'
        else:
            if not self._was_initialized:
                print 'Model does not exist yet. Use ArtmModel.initialize()/ArtmModel.fit_*()'
                return

            theta_matrix = self._master.GetThetaMatrix(self._model, clean_cache=remove_theta)
            document_ids = [item_id for item_id in theta_matrix.item_id]
            topic_names = [topic_name for topic_name in theta_matrix.topic_name]
            values = [[w for w in ws.value] for ws in theta_matrix.item_weights]
            retval = DataFrame(data=np.matrix(values).transpose(),
                               columns=document_ids,
                               index=topic_names)

            return retval

    def find_theta(self, batches=None, collection_name=None,
                   data_path='', data_format='batches'):
        """ ArtmModel.find_theta() --- find Theta matrix for new documents.

        Parameters:
        ----------
        - collection_name --- the name of text collection
          required if data_format == 'bow_uci').
          Is string, default = None

        - batches --- list of file names of batches to be processed.
          If not None, than data_format should be 'batches'.
          Is list of strings in format '*.batch', default = None

        - data_path --- 1) if data_format == 'batches' =>
          folder containing batches and dictionary
          2) if data_format == 'bow_uci' =>
          folder containing docword.txt and vocab.txt files
          3) if data_format == 'vowpal_wabbit' => file in Vowpal Wabbit format
          4) if data_format == 'plain_text' => file with text
          Is string, default = ''

        - data_format --- the type of input data:
          1) 'batches' --- the data in format of BigARTM
          2) 'bow_uci' --- Bag-Of-Words in UCI format
          3) 'vowpal_wabbit' --- Vowpal Wabbit format
          4) 'plain_text' --- source text
          Is string, default = 'batches'

        Returns:
        ----------
        - DataFrame (data, columns, rows), where:

          1) columns --- the ids of documents, for which the Theta
          matrix was requested

          2) rows --- the names of topics in topic model, that was
          used to create Theta

          3) data --- content of Theta matrix.
        """
        if collection_name is None and data_format == 'bow_uci':
            print 'No collection name was given, skip model.fit_offline()'

        if not data_format == 'batches' and batches is not None:
            print "batches != None require data_format == 'batches'"

        if not self._was_initialized:
            print 'Model does not exist yet. Use ArtmModel.initialize()/ArtmModel.fit_*()'
            return

        target_folder = data_path + '/batches_temp_' + str(random.uniform(0, 1))
        batches_list = []
        if data_format == 'batches':
            if batches is None:
                batches_list = glob.glob(data_path + "/*.batch")
                if len(batches_list) < 1:
                    print 'No batches were found, skip model.fit()'
                    return
                print 'find_theta() found ' + str(len(batches_list)) + ' batches'
            else:
                batches_list = [data_path + '/' + batch for batch in batches]

        elif data_format == 'bow_uci' or data_format == 'vowpal_wabbit':
            collection_parser_config = create_parser_config(data_path,
                                                            collection_name,
                                                            target_folder,
                                                            batch_size,
                                                            data_format)
            unique_tokens = library.Library().ParseCollection(collection_parser_config)
            batches_list = glob.glob(target_folder + "/*.batch")

        elif data_format == 'plain_text':
            raise NotImplementedError()
        else:
            print 'Unknown data format, skip model.find_theta()'

        results = self._master.ProcessBatches(
            pwt=self._model,
            batches=batches_list,
            target_nwt='nwt_hat',
            inner_iterations_count=self._num_document_passes,
            class_ids=self._class_ids,
            theta_matrix_type=library.ProcessBatchesArgs_ThetaMatrixType_Dense)

        theta_matrix = results.theta_matrix
        document_ids = [item_id for item_id in theta_matrix.item_id]
        topic_names = [topic_name for topic_name in theta_matrix.topic_name]
        values = [[w for w in ws.value] for ws in theta_matrix.item_weights]
        retval = DataFrame(data=np.matrix(values).transpose(),
                           columns=document_ids,
                           index=topic_names)

        # Remove temp batches folder if necessary
        if not data_format == 'batches':
            shutil.rmtree(target_folder)

        return retval

    def initialize(self, data_path=None, dictionary=None):
        """ ArtmModel.initialize() --- initialize topic model before learning.

        Parameters:
        ----------
        - data_path --- name of directory containing BigARTM batches.
          Is string, default = None

        - dictionary --- BigARTM collection dictionary.
          Is string, default = None

        Note:
        ----------
        Priority of initialization:
        1) batches in 'data_path'
        2) dictionary
        """
        if data_path is not None:
            self._master.InitializeModel(model_name=self._model,
                                         batch_folder=data_path,
                                         topics_count=self._topics_count,
                                         topic_names=self._topic_names)
        else:
            self._master.InitializeModel(model_name=self._model,
                                         dictionary=dictionary,
                                         topics_count=self._topics_count,
                                         topic_names=self._topic_names)

        args = messages_pb2.GetTopicModelArgs()
        args.request_type = library.GetTopicModelArgs_RequestType_TopicNames
        topic_model = self._master.GetTopicModel(model=self._model, args=args)
        self._topic_names = [topic_name for topic_name in topic_model.topic_name]
        self._was_initialized = True

        # Remove all info about previous iterations
        self._scores_info = {}
        self._synchronizations_processed = 0

    def visualize(self, num_top_tokens=30, dictionary_path=None, lambda_step=0.1):
        """ ArtmModel.visualize() --- visualize topic model after learning.

        Parameters:
        ----------
        - num_top_tokens --- number of top tokens to be used in visualization.
          Is int, default = 30

        - dictionary_path --- path to file containing BigARTM
          collection dictionary.
          Is string, default = None

        - lambda_step ---the parameter of the LDAvis visualizer.
          Is double in (0, 1), default = 0.1

        Returns:
        ---------
        - an object of visualization.TopicModelVisualization() class

        Note:
        ----------
        This method still works incorrectly and dramatically ineffective while
        trying to visualize collection with huge number of topics and tokens.
        It will be complete in next release.
        """
        if not self._was_initialized:
            print 'Model does not exist yet. Use ArtmModel.initialize()/ArtmModel.fit_*()'
            return

        dictionary = messages_pb2.DictionaryConfig()
        if dictionary_path is not None:
            dictionary = library.Library().LoadDictionary(dictionary_path)
        else:
            print 'dictionary path is None, skip visualization.'
            return

        if not os.path.exists('../artm/_js/ldavis.js'):
            download_ldavis()

        p_wt_model = self._master.GetTopicModel(self._model)

        phi = np.matrix([token_w for token_w in
                        [token_weights.value for token_weights in p_wt_model.token_weights]])

        vocab = np.matrix([token for token in p_wt_model.token])
        token_to_index = {}
        for i in range(len(p_wt_model.token)):
            token_to_index[p_wt_model.token[i]] = i

        dist_matrix = sp_dist.pdist(phi.transpose(), lambda u, v: sym_kl_dist(u, v))
        pca_model = sklearn.decomposition.pca.PCA(2)
        centers = pca_model.fit_transform(sp_dist.squareform(dist_matrix)).transpose()

        topic_proportion = [1.0 / self._topics_count] * self._topics_count

        term_frequency = np.matrix([entry.token_count * 1.0
                                   for entry in dictionary.entry]).transpose()
        term_proportion = np.matrix(np.divide(term_frequency, sum(term_frequency)))

        term_topic_frequency = phi
        p_w = np.matrix(np.sum(phi, axis=1))
        term_topic_frequency = np.multiply(term_topic_frequency, (np.divide(term_frequency,
                                                                            p_w + GLOB_EPS)))

        topic_given_term = np.matrix(np.divide(phi, p_w + GLOB_EPS))
        vect_log = np.vectorize(lambda x: math.log(x) if x > GLOB_EPS else 0)
        kernel = np.matrix(np.multiply(topic_given_term, vect_log(topic_given_term)))
        saliency = np.matrix(np.multiply(term_proportion, np.matrix(np.sum(kernel, axis=1))))

        sorting_indices = saliency.ravel().argsort()
        default_terms = np.matrix(vocab[0, sorting_indices][0, 0: num_top_tokens])

        counts = np.matrix(term_frequency.transpose()[0, sorting_indices])
        rs = np.matrix(range(0, num_top_tokens)[::-1])
        topic_str_list = ['Topic' + str(i) for i in range(1, self._topics_count + 1)]
        category = [x for item in topic_str_list for x in itertools.repeat(item, num_top_tokens)]
        topics = [x for item in range(self._topics_count)
                  for x in itertools.repeat(item, num_top_tokens)]

        lift = np.divide(phi, term_proportion + GLOB_EPS)
        phi_column = phi.reshape(phi.size, 1)
        lift_column = lift.reshape(lift.size, 1)

        tinfo = {}
        tinfo['Term'] = np.array(default_terms)[0].tolist()
        tinfo['Category'] = ['Default' for _ in np.array(default_terms)[0].tolist()]
        tinfo['logprob'] = np.array(rs)[0].tolist()
        tinfo['loglift'] = np.array(rs)[0].tolist()
        tinfo['Freq'] = np.array(counts)[0].tolist()
        tinfo['Total'] = np.array(counts)[0].tolist()

        term_indices = []
        topic_indices = []

        def find_relevance(i, term_indices, topic_indices, tinfo):
            relevance = np.matrix(i * vect_log(phi) + (1 - i) * vect_log(lift))
            idx = np.matrix(
                np.apply_along_axis(lambda x: x.ravel().argsort()[range(0, num_top_tokens)],
                                    axis=0, arr=relevance))
            idx.resize(1, idx.size)
            indices = np.concatenate((idx,
                                      np.matrix([x for i in range(self._topics_count)
                                                 for x in itertools.repeat(i, num_top_tokens)])),
                                     axis=1)

            tinfo['Term'] += np.array(vocab[0, idx])[0].tolist()
            tinfo['Category'] += category
            tinfo['logprob'] += np.array(
                                    np.round(vect_log(phi_column[indices, 0]), 4))[0].tolist()
            tinfo['loglift'] += np.array(
                                    np.round(vect_log(lift_column[indices, 0]), 4))[0].tolist()
            term_indices += np.array(idx)[0].tolist()
            topic_indices += topics

        for i in np.arange(0, 1, lambda_step):
            find_relevance(i, term_indices, topic_indices, tinfo)

        tinfo['Total'] += np.array(term_frequency[term_indices, 0])[0].tolist()
        for i in range(len(term_indices)):
            tinfo['Freq'].append(term_topic_frequency[term_indices[i], topic_indices[i]])

        # ut = list(set(tinfo['Term']))
        # ut.sort()
        # m = [token_to_index[token] for token in ut]
        # m.sort()
        # phi_submatrix = np.matrix(term_topic_frequency[m, :])

        all_tokens = []
        all_topics = []
        all_values = []
        for token_index in range(len(p_wt_model.token_weights)):
            for topic_index in range(self._topics_count):
                all_tokens.append(p_wt_model.token[token_index])
                all_topics.append(topic_index + 1)
                all_values.append(p_wt_model.token_weights[token_index].value[topic_index])

        data = {'mdsDat': {'x': list(centers[0]),
                           'y': list(centers[1]),
                           'topics': range(1, self._topics_count + 1),
                           'Freq': [i * 100 for i in topic_proportion],
                           'cluster': [1] * self._topics_count},
                'tinfo': {'Term': tinfo['Term'],
                          'logprob': tinfo['logprob'],
                          'loglift': tinfo['loglift'],
                          'Freq': tinfo['Freq'],
                          'Total': tinfo['Total'],
                          'Category': tinfo['Category']},
                'token.table': {'Term': all_tokens,
                                'Topic': all_topics,
                                'Freq': all_values},
                'R': num_top_tokens,
                'lambda.step': lambda_step,
                'plot.opts': {'xlab': 'PC-1',
                              'ylab': 'PC-2'},
                'topic_order': [i for i in range(self._topics_count)]}

        file_name = 'lda.json'
        if os.path.isfile(file_name):
            os.remove(file_name)

        with open(file_name, "w") as outfile:
            json.dump(data, outfile, indent=2)
        return visualization.TopicModelVisualization(data)
