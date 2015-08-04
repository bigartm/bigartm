import uuid
import random

import messages_pb2
import library as lib


__all__ = [
    'SmoothSparsePhiRegularizer',
    'SmoothSparseThetaRegularizer',
    'DecorrelatorPhiRegularizer',
    'LabelRegularizationPhiRegularizer',
    'SpecifiedSparsePhiRegularizer',
    'ImproveCoherencePhiRegularizer'
]

CREATE_REGULARIZER_CONFIG = {
    lib.RegularizerConfig_Type_SmoothSparsePhi: messages_pb2.SmoothSparsePhiConfig,
    lib.RegularizerConfig_Type_SmoothSparseTheta: messages_pb2.SmoothSparseThetaConfig,
    lib.RegularizerConfig_Type_DecorrelatorPhi: messages_pb2.DecorrelatorPhiConfig,
    lib.RegularizerConfig_Type_LabelRegularizationPhi: messages_pb2.LabelRegularizationPhiConfig,
    lib.RegularizerConfig_Type_SpecifiedSparsePhi: messages_pb2.SpecifiedSparsePhiConfig,
    lib.RegularizerConfig_Type_ImproveCoherencePhi: messages_pb2.ImproveCoherencePhiConfig
}


def _reconfigure_field(obj, field, field_name):
    setattr(obj, '_' + field_name, field)
    config = CREATE_REGULARIZER_CONFIG[obj._type]()
    config.CopyFrom(obj._config)
    if isinstance(field, list):
        config.ClearField(field_name)
        for value in field:
            getattr(config, field_name).append(value)
    else:
        setattr(config, field_name, field)
    obj.regularizer.Reconfigure(obj.regularizer.config_.type, config)


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


class BaseRegularizer(object):
    def __init__(self, name, tau, topic_names):
        config = CREATE_REGULARIZER_CONFIG[self._type]()
        self._topic_names = []

        if name is None:
            name = self._type + ': ' + uuid.uuid1().urn
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
                self._topic_names.append(topic_name)

        self._name = name
        self.tau = tau
        self._config = config
        self._regularizer = None  # Reserve place for the regularizer

    @property
    def name(self):
        return self._name

    @property
    def topic_names(self):
        return self._topic_names

    @property
    def regularizer(self):
        return self._regularizer

    @property
    def config(self):
        return self._config

    @property
    def type(self):
        return self._type

    @topic_names.setter
    def topic_names(self, topic_names):
        _reconfigure_field(self, topic_names, 'topic_names')


class BaseRegularizerPhi(BaseRegularizer):
    def __init__(self, name, tau, topic_names,
                 class_ids, dictionary_name):
        BaseRegularizer.__init__(self,
                                 name=name,
                                 tau=tau,
                                 topic_names=topic_names)
        self._class_ids = []
        self._dictionary_name = ''

        if class_ids is not None:
            self._config.ClearField('class_id')
            for class_id in class_ids:
                self._config.class_id.append(class_id)
                self._class_ids.append(class_id)
        if dictionary_name is not None:
            self._config.dictionary_name = dictionary_name
            self._dictionary_name = dictionary_name

    @property
    def class_ids(self):
        return self._class_ids

    @property
    def dictionary_name(self):
        return self._dictionary_name

    @class_ids.setter
    def class_ids(self, class_ids):
        _reconfigure_field(self, class_ids, 'class_ids')

    @dictionary_name.setter
    def dictionary_name(self, dictionary_name):
        _reconfigure_field(self, dictionary_name, 'dictionary_name')


class BaseRegularizerTheta(BaseRegularizer):
    def __init__(self, name, tau, topic_names, alpha_iter):
        BaseRegularizer.__init__(self,
                                 name=name,
                                 tau=tau,
                                 topic_names=topic_names)
        self._alpha_iter = []

        if alpha_iter is not None:
            self._config.ClearField('alpha_iter')
            for alpha in alpha_iter:
                self._config.alpha_iter.append(alpha)
                self._alpha_iter.append(alpha)

    @property
    def alpha_iter(self):
        return self._alpha_iter

    @alpha_iter.setter
    def alpha_iter(self, alpha_iter):
        _reconfigure_field(self, alpha_iter, 'alpha_iter')


###################################################################################################
# SECTION OF REGULARIZER CLASSES
###################################################################################################
class SmoothSparsePhiRegularizer(BaseRegularizerPhi):
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

    _type = lib.RegularizerConfig_Type_SmoothSparsePhi

    def __init__(self, name=None, tau=1.0, class_ids=None,
                 topic_names=None, dictionary_name=None):
        BaseRegularizerPhi.__init__(self,
                                    name=name,
                                    tau=tau,
                                    topic_names=topic_names,
                                    class_ids=class_ids,
                                    dictionary_name=dictionary_name)


class SmoothSparseThetaRegularizer(BaseRegularizerTheta):
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

    _type = lib.RegularizerConfig_Type_SmoothSparseTheta

    def __init__(self, name=None, tau=1.0, topic_names=None, alpha_iter=None):
        BaseRegularizerTheta.__init__(self,
                                      name=name,
                                      tau=tau,
                                      topic_names=topic_names,
                                      alpha_iter=alpha_iter)


class DecorrelatorPhiRegularizer(BaseRegularizerPhi):
    """DecorrelatorPhiRegularizer is a regularizer in ArtmModel (public class)

    Args:
      name (str): the identifier of regularizer, will be auto-generated if not specified
      tau (double): the coefficient of regularization for this regularizer, default=1.0
      class_ids (list of str): list of class_ids to regularize, will regularize all
      classes if not specified
      topic_names (list of str): list of names of topics to regularize, will regularize
      all topics if not specified
    """

    _type = lib.RegularizerConfig_Type_DecorrelatorPhi

    def __init__(self, name=None, tau=1.0, class_ids=None, topic_names=None):
        BaseRegularizerPhi.__init__(self,
                                    name=name,
                                    tau=tau,
                                    topic_names=topic_names,
                                    class_ids=class_ids,
                                    dictionary_name=None)

    @property
    def dictionary_name(self):
        raise KeyError('No dictionary_name parameter')

    @dictionary_name.setter
    def dictionary_name(self, dictionary_name):
        raise KeyError('No dictionary_name parameter')


class LabelRegularizationPhiRegularizer(BaseRegularizerPhi):
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

    _type = lib.RegularizerConfig_Type_LabelRegularizationPhi

    def __init__(self, name=None, tau=1.0, class_ids=None,
                 topic_names=None, dictionary_name=None):
        BaseRegularizerPhi.__init__(self,
                                    name=name,
                                    tau=tau,
                                    topic_names=topic_names,
                                    class_ids=class_ids,
                                    dictionary_name=dictionary_name)


class SpecifiedSparsePhiRegularizer(BaseRegularizerPhi):
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

    _type = lib.RegularizerConfig_Type_SpecifiedSparsePhi

    def __init__(self, name=None, tau=1.0, topic_names=None, class_id=None,
                 num_max_elements=None, probability_threshold=None, sparse_by_columns=True):
        BaseRegularizerPhi.__init__(self,
                                    name=name,
                                    tau=tau,
                                    topic_names=topic_names,
                                    dictionary_name=None,
                                    class_ids=None)
        self._class_id = '@default_class'
        self._num_max_elements = 20
        self._probability_threshold = 0.99
        self._sparse_by_columns = True

        if class_id is not None:
            self._config.class_id = class_id
            self._class_id = class_id
        if num_max_elements is not None:
            self._config.max_elements_count = num_max_elements
            self._num_max_elements = num_max_elements
        if probability_threshold is not None:
            self._config.probability_threshold = probability_threshold
            self._probability_threshold = probability_threshold
        if sparse_by_columns is not None:
            if sparse_by_columns is True:
                self._config.mode = lib.SpecifiedSparsePhiConfig_Mode_SparseTopics
                self._sparse_by_columns = True
            else:
                self._config.mode = lib.SpecifiedSparsePhiConfig_Mode_SparseTokens
                self._sparse_by_columns = False

    @property
    def class_id(self):
        return self._class_id

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
    def class_ids(self):
        raise KeyError('No class_ids parameter')

    @property
    def dictionary_name(self):
        raise KeyError('No dictionary_name parameter')

    @class_id.setter
    def class_id(self, class_id):
        _reconfigure_field(self, class_id, 'class_id')

    @num_max_elements.setter
    def num_max_elements(self, num_max_elements):
        self._num_max_elements = num_max_elements
        config = messages_pb2.SpecifiedSparseRegularizationPhiConfig()
        config.CopyFrom(self._config)
        config.max_elements_count = num_max_elements
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @probability_threshold.setter
    def probability_threshold(self, probability_threshold):
        _reconfigure_field(self, probability_threshold, 'probability_threshold')

    @sparse_by_columns.setter
    def sparse_by_columns(self, sparse_by_columns):
        self._sparse_by_columns = sparse_by_columns
        config = messages_pb2.SpecifiedSparseRegularizationPhiConfig()
        config.CopyFrom(self._config)
        if sparse_by_columns is True:
            config.mode = lib.SpecifiedSparsePhiConfig_Mode_SparseTopics
        else:
            config.mode = lib.SpecifiedSparsePhiConfig_Mode_SparseTokens
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @class_ids.setter
    def class_ids(self, class_ids):
        raise KeyError('No class_ids parameter')

    @dictionary_name.setter
    def dictionary_name(self, dictionary_name):
        raise KeyError('No dictionary_name parameter')


class ImproveCoherencePhiRegularizer(BaseRegularizerPhi):
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

    _type = lib.RegularizerConfig_Type_ImproveCoherencePhi

    def __init__(self, name=None, tau=1.0, class_ids=None,
                 topic_names=None, dictionary_name=None):
        BaseRegularizerPhi.__init__(self,
                                    name=name,
                                    tau=tau,
                                    topic_names=topic_names,
                                    class_ids=class_ids,
                                    dictionary_name=dictionary_name)
