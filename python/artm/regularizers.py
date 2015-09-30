import uuid
import random

from wrapper import messages_pb2 as messages
from wrapper import constants as const


__all__ = [
    'SmoothSparsePhiRegularizer',
    'SmoothSparseThetaRegularizer',
    'DecorrelatorPhiRegularizer',
    'LabelRegularizationPhiRegularizer',
    'SpecifiedSparsePhiRegularizer',
    'ImproveCoherencePhiRegularizer'
]


def _reconfigure_field(obj, field, field_name, proto_field_name=None):
    if proto_field_name is None:
        proto_field_name = field_name
    setattr(obj, '_{0}'.format(field_name), field)

    config = obj._config_message()
    config.CopyFrom(obj._config)
    if isinstance(field, list):
        config.ClearField(proto_field_name)
        for value in field:
            getattr(config, proto_field_name).append(value)
    else:
        setattr(config, proto_field_name, field)
    obj._master.reconfigure_regularizer(obj.name, obj.type, config)


class Regularizers(object):
    """Regularizers represents a storage of regularizers in ArtmModel (private class)

    Args:
      master (reference): reference to MasterComponent object, no default
    """

    def __init__(self, master):
        self._data = {}
        self._master = master

    def add(self, regularizer):
        """Regularizers.add() --- add regularizer into ArtmModel

        Args:
          regularizer: an object of ***Regularizer class, no default
        """
        if regularizer.name in self._data:
            raise ValueError('Regularizer with name {0} is already exist'.format(regularizer.name))
        else:
            self._master.create_regularizer(regularizer.name, regularizer.type, regularizer.config)
            regularizer._master = self._master
            self._data[regularizer.name] = regularizer

    def __getitem__(self, name):
        """Regularizers.__getitem__() --- get regularizer with given name

        Args:
          name (str): name of the regularizer, no default
        """
        if name in self._data:
            return self._data[name]
        else:
            raise KeyError('No regularizer with name {0}'.format(name))

    @property
    def data(self):
        return self._data


class BaseRegularizer(object):

    _config_message = None

    def __init__(self, name, tau, topic_names):
        if self._config_message is None:
            raise NotImplementedError()
        config = self._config_message()

        if name is None:
            name = '{0}:{1}'.format(self._type, uuid.uuid1().urn)

        self._topic_names = []
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
                self._topic_names.append(topic_name)

        self._name = name
        self.tau = tau
        self._config = config
        self._master = None  # reserve place for master

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
        if class_ids is not None:
            self._config.ClearField('class_id')
            for class_id in class_ids:
                self._config.class_id.append(class_id)
                self._class_ids.append(class_id)

        self._dictionary_name = ''
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

    _config_message = messages.SmoothSparsePhiConfig
    _type = const.RegularizerConfig_Type_SmoothSparsePhi

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

    _config_message = messages.SmoothSparseThetaConfig
    _type = const.RegularizerConfig_Type_SmoothSparseTheta

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

    _config_message = messages.DecorrelatorPhiConfig
    _type = const.RegularizerConfig_Type_DecorrelatorPhi

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

    _config_message = messages.LabelRegularizationPhiConfig
    _type = const.RegularizerConfig_Type_LabelRegularizationPhi

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

    _config_message = messages.SpecifiedSparsePhiConfig
    _type = const.RegularizerConfig_Type_SpecifiedSparsePhi

    def __init__(self, name=None, tau=1.0, topic_names=None, class_id=None,
                 num_max_elements=None, probability_threshold=None, sparse_by_columns=True):
        BaseRegularizerPhi.__init__(self,
                                    name=name,
                                    tau=tau,
                                    topic_names=topic_names,
                                    dictionary_name=None,
                                    class_ids=None)

        self._class_id = '@default_class'
        if class_id is not None:
            self._config.class_id = class_id
            self._class_id = class_id

        self._num_max_elements = 20
        if num_max_elements is not None:
            self._config.max_elements_count = num_max_elements
            self._num_max_elements = num_max_elements

        self._probability_threshold = 0.99
        if probability_threshold is not None:
            self._config.probability_threshold = probability_threshold
            self._probability_threshold = probability_threshold

        self._sparse_by_columns = True
        if sparse_by_columns is not None:
            if sparse_by_columns is True:
                self._config.mode = const.SpecifiedSparsePhiConfig_Mode_SparseTopics
                self._sparse_by_columns = True
            else:
                self._config.mode = const.SpecifiedSparsePhiConfig_Mode_SparseTokens
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
        _reconfigure_field(self, num_max_elements, 'num_max_elements', 'max_elements_count')

    @probability_threshold.setter
    def probability_threshold(self, probability_threshold):
        _reconfigure_field(self, probability_threshold, 'probability_threshold')

    @sparse_by_columns.setter
    def sparse_by_columns(self, sparse_by_columns):
        self._sparse_by_columns = sparse_by_columns
        config = messages.SpecifiedSparseRegularizationPhiConfig()
        config.CopyFrom(self._config)
        if sparse_by_columns is True:
            config.mode = const.SpecifiedSparsePhiConfig_Mode_SparseTopics
        else:
            config.mode = const.SpecifiedSparsePhiConfig_Mode_SparseTokens
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

    _config_message = messages.ImproveCoherencePhiConfig
    _type = const.RegularizerConfig_Type_ImproveCoherencePhi

    def __init__(self, name=None, tau=1.0, class_ids=None,
                 topic_names=None, dictionary_name=None):
        BaseRegularizerPhi.__init__(self,
                                    name=name,
                                    tau=tau,
                                    topic_names=topic_names,
                                    class_ids=class_ids,
                                    dictionary_name=dictionary_name)
