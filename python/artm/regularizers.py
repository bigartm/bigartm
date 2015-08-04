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


THETA_REGULARIZER_TYPE = 0
PHI_REGULARIZER_TYPE = 1

__all__ = [
    'SmoothSparsePhiRegularizer',
    'SmoothSparseThetaRegularizer',
    'DecorrelatorPhiRegularizer',
    'LabelRegularizationPhiRegularizer',
    'SpecifiedSparsePhiRegularizer',
    'ImproveCoherencePhiRegularizer'
]


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
