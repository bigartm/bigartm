import uuid
import random

from . import wrapper
from wrapper import messages_pb2 as messages
from wrapper import constants as const


__all__ = [
    'KlFunctionInfo',
    'SmoothSparsePhiRegularizer',
    'SmoothSparseThetaRegularizer',
    'DecorrelatorPhiRegularizer',
    'LabelRegularizationPhiRegularizer',
    'SpecifiedSparsePhiRegularizer',
    'ImproveCoherencePhiRegularizer',
    'SmoothPtdwRegularizer',
    'TopicSelectionThetaRegularizer',
    'BitermsPhiRegularizer',
	'HierarchySparsingThetaRegularizer'
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
    obj._master.reconfigure_regularizer(obj.name, obj.config, obj.tau, obj.gamma)


class KlFunctionInfo(object):
    def __init__(self, function_type='log', power_value=2.0):
        """
        :param str function_type: the type of function, 'log' (logarithm) or 'pol' (polynomial)
        :param float power_value: the double power of polynomial, ignored if type = 'log'
        """
        if function_type not in ['log', 'pol']:
            raise ValueError('Function type can be only "log" or "pol"')

        self.function_type = function_type
        self.power_value = power_value

    def _update_config(self, obj, first=False):
        config = obj._config_message()
        config.CopyFrom(obj._config)

        if self.function_type == 'log':
            config.transform_config.type = const.TransformConfig_TransformType_Constant
        elif self.function_type == 'pol':
            config.transform_config.type = const.TransformConfig_TransformType_Polynomial
            config.transform_config.n = self.power_value  # power_value - 1, but *x gives no change
            config.transform_config.a = self.power_value

        obj._config = config
        if not first:
            obj._master.reconfigure_regularizer(obj.name, obj.config, obj.tau, obj.gamma)


class Regularizers(object):
    def __init__(self, master):
        self._data = {}
        self._master = master

    def add(self, regularizer):
        if regularizer.name not in self._data:
            self._master.create_regularizer(regularizer.name, regularizer.config, regularizer.tau, regularizer.gamma)
            regularizer._master = self._master
            self._data[regularizer.name] = regularizer

    def __getitem__(self, name):
        if name in self._data:
            return self._data[name]
        else:
            raise KeyError('No regularizer with name {0}'.format(name))

    def size(self):
        return len(self._data.keys())

    @property
    def data(self):
        return self._data


class BaseRegularizer(object):
    _config_message = None

    def __init__(self, name, tau, gamma, config):
        if self._config_message is None:
            raise NotImplementedError()

        if name is None:
            name = '{0}:{1}'.format(self._type, uuid.uuid1().urn)

        self._name = name
        self._tau = tau
        self._gamma = gamma
        self._config = config if config is not None else self._config_message()
        self._master = None  # reserve place for master

    @property
    def name(self):
        return self._name

    @property
    def tau(self):
        return self._tau

    @property
    def gamma(self):
        return self._gamma

    @property
    def regularizer(self):
        return self._regularizer

    @property
    def config(self):
        return self._config

    @property
    def type(self):
        return self._type

    @name.setter
    def name(self, name):
        raise RuntimeError("It's impossible to change regularizer name")

    @tau.setter
    def tau(self, tau):
        self._tau = tau
        self._master.reconfigure_regularizer(self._name, self._config, tau, self._gamma)

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma
        self._master.reconfigure_regularizer(self._name, self._config, self._tau, gamma)

    @config.setter
    def config(self, config):
        self._config = config
        self._master.reconfigure_regularizer(self._name, config, self._tau, self._gamma)


class BaseRegularizerPhi(BaseRegularizer):
    def __init__(self, name, tau, gamma, config, topic_names, class_ids, dictionary):
        BaseRegularizer.__init__(self,
                                 name=name,
                                 tau=tau,
                                 gamma=gamma,
                                 config=config)

        self._class_ids = []
        if class_ids is not None:
            self._config.ClearField('class_id')
            for class_id in class_ids:
                self._config.class_id.append(class_id)
                self._class_ids.append(class_id)

        self._topic_names = []
        if topic_names is not None:
            self._config.ClearField('topic_name')
            for topic_name in topic_names:
                self._config.topic_name.append(topic_name)
                self._topic_names.append(topic_name)

        self._dictionary_name = ''
        if dictionary is not None:
            dictionary_name = dictionary if isinstance(dictionary, str) else dictionary.name
            self._config.dictionary_name = dictionary_name
            self._dictionary_name = dictionary_name

    @property
    def class_ids(self):
        return self._class_ids

    @property
    def topic_names(self):
        return self._topic_names

    @property
    def dictionary(self):
        return self._dictionary_name

    @class_ids.setter
    def class_ids(self, class_ids):
        _reconfigure_field(self, class_ids, 'class_ids')

    @dictionary.setter
    def dictionary(self, dictionary):
        dictionary_name = dictionary if isinstance(dictionary, str) else dictionary.name
        _reconfigure_field(self, dictionary_name, 'dictionary_name')

    @topic_names.setter
    def topic_names(self, topic_names):
        _reconfigure_field(self, topic_names, 'topic_names')


class BaseRegularizerTheta(BaseRegularizer):
    def __init__(self, name, tau, config, topic_names, alpha_iter):
        BaseRegularizer.__init__(self,
                                 name=name,
                                 tau=tau,
                                 gamma=None,
                                 config=config)
        self._alpha_iter = []
        if alpha_iter is not None:
            self._config.ClearField('alpha_iter')
            for alpha in alpha_iter:
                self._config.alpha_iter.append(alpha)
                self._alpha_iter.append(alpha)

        self._topic_names = []
        if topic_names is not None:
            self._config.ClearField('topic_name')
            for topic_name in topic_names:
                self._config.topic_name.append(topic_name)
                self._topic_names.append(topic_name)

    @property
    def alpha_iter(self):
        return self._alpha_iter

    @property
    def topic_names(self):
        return self._topic_names

    @alpha_iter.setter
    def alpha_iter(self, alpha_iter):
        _reconfigure_field(self, alpha_iter, 'alpha_iter')

    @topic_names.setter
    def topic_names(self, topic_names):
        _reconfigure_field(self, topic_names, 'topic_names')


###################################################################################################
# SECTION OF REGULARIZER CLASSES
###################################################################################################
class SmoothSparsePhiRegularizer(BaseRegularizerPhi):
    _config_message = messages.SmoothSparsePhiConfig
    _type = const.RegularizerType_SmoothSparsePhi

    def __init__(self, name=None, tau=1.0, gamma=None, class_ids=None, topic_names=None,
                 dictionary=None, kl_function_info=None, config=None):
        """
        :param str name: the identifier of regularizer, will be auto-generated if not specified
        :param float tau: the coefficient of regularization for this regularizer
        :param float gamma: the coefficient of relative regularization for this regularizer
        :param class_ids: list of class_ids to regularize, will\
                                     regularize all classes if not specified
        :type class_ids: list of str
        :param topic_names: list of names of topics to regularize,\
                                     will regularize all topics if not specified
        :type topic_names: list of str
        :param dictionary: BigARTM collection dictionary,\
                                     won't use dictionary if not specified
        :type dictionary: str or reference to Dictionary object
        :param kl_function_info: class with additional info about\
                                     function under KL-div in regularizer
        :type kl_function_info: KlFunctionInfo object
        :param config: the low-level config of this regularizer
        :type config: protobuf object
        """
        BaseRegularizerPhi.__init__(self,
                                    name=name,
                                    tau=tau,
                                    gamma=gamma,
                                    config=config,
                                    topic_names=topic_names,
                                    class_ids=class_ids,
                                    dictionary=dictionary)

        self._kl_function_info = KlFunctionInfo()
        if kl_function_info is not None:
            self._kl_function_info = kl_function_info
        self._kl_function_info._update_config(self, first=True)

    @property
    def kl_function_info(self):
        return self._kl_function_info

    @kl_function_info.setter
    def kl_function_info(self, kl_function_info):
        self._kl_function_info = kl_function_info
        kl_function_info._update_config(self)


class SmoothSparseThetaRegularizer(BaseRegularizerTheta):
    _config_message = messages.SmoothSparseThetaConfig
    _type = const.RegularizerType_SmoothSparseTheta

    def __init__(self, name=None, tau=1.0, topic_names=None,
                 alpha_iter=None, kl_function_info=None, config=None):
        """
        :param str name: the identifier of regularizer, will be auto-generated if not specified
        :param float tau: the coefficient of regularization for this regularizer
        :param alpha_iter: list of additional coefficients of regularization on each iteration\
                           over document. Should have length equal to model.num_document_passes
        :type alpha_iter: list of str
        :param topic_names: list of names of topics to regularize,\
                                     will regularize all topics if not specified
        :type topic_names: list of str
        :param kl_function_info: class with additional info about\
                                     function under KL-div in regularizer
        :type kl_function_info: KlFunctionInfo object
        :param config: the low-level config of this regularizer
        :type config: protobuf object
        """
        BaseRegularizerTheta.__init__(self,
                                      name=name,
                                      tau=tau,
                                      config=config,
                                      topic_names=topic_names,
                                      alpha_iter=alpha_iter)

        self._kl_function_info = KlFunctionInfo()
        if kl_function_info is not None:
            self._kl_function_info = kl_function_info
        self._kl_function_info._update_config(self, first=True)

    @property
    def kl_function_info(self):
        return self._kl_function_info

    @kl_function_info.setter
    def kl_function_info(self, kl_function_info):
        self._kl_function_info = kl_function_info
        kl_function_info._update_config(self)


class DecorrelatorPhiRegularizer(BaseRegularizerPhi):
    _config_message = messages.DecorrelatorPhiConfig
    _type = const.RegularizerType_DecorrelatorPhi

    def __init__(self, name=None, tau=1.0, gamma=None, class_ids=None, topic_names=None, config=None):
        """
        :param str name: the identifier of regularizer, will be auto-generated if not specified
        :param float tau: the coefficient of regularization for this regularizer
        :param float gamma: the coefficient of relative regularization for this regularizer
        :param class_ids: list of class_ids to regularize, will\
                                     regularize all classes if not specified
        :type class_ids: list of str
        :param topic_names: list of names of topics to regularize,\
                                     will regularize all topics if not specified
        :type topic_names: list of str
        :param config: the low-level config of this regularizer
        :type config: protobuf object
        """
        BaseRegularizerPhi.__init__(self,
                                    name=name,
                                    tau=tau,
                                    gamma=gamma,
                                    config=config,
                                    topic_names=topic_names,
                                    class_ids=class_ids,
                                    dictionary=None)

    @property
    def dictionary(self):
        raise KeyError('No dictionary parameter')

    @dictionary.setter
    def dictionary(self, dictionary):
        raise KeyError('No dictionary parameter')


class LabelRegularizationPhiRegularizer(BaseRegularizerPhi):
    _config_message = messages.LabelRegularizationPhiConfig
    _type = const.RegularizerType_LabelRegularizationPhi

    def __init__(self, name=None, tau=1.0, gamma=None, class_ids=None,
                 topic_names=None, dictionary=None, config=None):
        """
        :param str name: the identifier of regularizer, will be auto-generated if not specified
        :param float tau: the coefficient of regularization for this regularizer
        :param float gamma: the coefficient of relative regularization for this regularizer
        :param class_ids: list of class_ids to regularize, will\
                                     regularize all classes if not specified
        :type class_ids: list of str
        :param topic_names: list of names of topics to regularize,\
                                     will regularize all topics if not specified
        :type topic_names: list of str
        :param dictionary: BigARTM collection dictionary,\
                                     won't use dictionary if not specified
        :type dictionary: str or reference to Dictionary object
        :param config: the low-level config of this regularizer
        :type config: protobuf object
        """
        BaseRegularizerPhi.__init__(self,
                                    name=name,
                                    tau=tau,
                                    gamma=gamma,
                                    config=config,
                                    topic_names=topic_names,
                                    class_ids=class_ids,
                                    dictionary=dictionary)


class SpecifiedSparsePhiRegularizer(BaseRegularizerPhi):
    _config_message = messages.SpecifiedSparsePhiConfig
    _type = const.RegularizerType_SpecifiedSparsePhi

    def __init__(self, name=None, tau=1.0, gamma=None, topic_names=None, class_id=None,
                 num_max_elements=None, probability_threshold=None, sparse_by_columns=True, config=None):
        """
        :param str name: the identifier of regularizer, will be auto-generated if not specified
        :param float tau: the coefficient of regularization for this regularizer
        :param float gamma: the coefficient of relative regularization for this regularizer
        :param class_id: class_id to regularize
        :param topic_names: list of names of topics to regularize,\
                                     will regularize all topics if not specified
        :type topic_names: list of str
        :param int num_max_elements: number of elements to save in row/column
        :param float probability_threshold: if m elements in row/column sum into value >=\
                                     probability_threshold, m < n => only these elements would\
                                     be saved. Value should be in (0, 1), default=None
        :param bool sparse_by_columns: find max elements in column or in row
        :param config: the low-level config of this regularizer
        :type config: protobuf object
        """
        BaseRegularizerPhi.__init__(self,
                                    name=name,
                                    tau=tau,
                                    gamma=gamma,
                                    config=config,
                                    topic_names=topic_names,
                                    dictionary=None,
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
                self._config.mode = const.SpecifiedSparsePhiConfig_SparseMode_SparseTopics
                self._sparse_by_columns = True
            else:
                self._config.mode = const.SpecifiedSparsePhiConfig_SparseMode_SparseTokens
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
    def dictionary(self):
        raise KeyError('No dictionary parameter')

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
            config.mode = const.SpecifiedSparsePhiConfig_SparseMode_SparseTopics
        else:
            config.mode = const.SpecifiedSparsePhiConfig_SparseMode_SparseTokens
        self.regularizer.Reconfigure(self.regularizer.config_.type, config)

    @class_ids.setter
    def class_ids(self, class_ids):
        raise KeyError('No class_ids parameter')

    @dictionary.setter
    def dictionary(self, dictionary):
        raise KeyError('No dictionary parameter')


class ImproveCoherencePhiRegularizer(BaseRegularizerPhi):
    _config_message = messages.ImproveCoherencePhiConfig
    _type = const.RegularizerType_ImproveCoherencePhi

    def __init__(self, name=None, tau=1.0, gamma=None, class_ids=None,
                 topic_names=None, dictionary=None, config=None):
        """
        :param str name: the identifier of regularizer, will be auto-generated if not specified
        :param float tau: the coefficient of regularization for this regularizer
        :param float gamma: the coefficient of relative regularization for this regularizer
        :param class_ids: list of class_ids to regularize, will\
                                     regularize all classes if not specified,
                                     dictionaty should contain pairwise tokens coocurancy info
        :type class_ids: list of str
        :param topic_names: list of names of topics to regularize,\
                                     will regularize all topics if not specified
        :type topic_names: list of str
        :param dictionary: BigARTM collection dictionary, won't use dictionary if not\
                                     specified, in this case regularizer is useless
        :type dictionary: str or reference to Dictionary object
        :param config: the low-level config of this regularizer
        :type config: protobuf object
        """
        BaseRegularizerPhi.__init__(self,
                                    name=name,
                                    tau=tau,
                                    gamma=gamma,
                                    config=config,
                                    topic_names=topic_names,
                                    class_ids=class_ids,
                                    dictionary=dictionary)


class SmoothPtdwRegularizer(BaseRegularizer):
    _config_message = messages.SmoothPtdwConfig
    _type = const.RegularizerType_SmoothPtdw

    def __init__(self, name=None, tau=1.0, config=None):
        """
        :param str name: the identifier of regularizer, will be auto-generated if not specified
        :param float tau: the coefficient of regularization for this regularizer
        :param config: the low-level config of this regularizer
        :type config: protobuf object
        """
        BaseRegularizer.__init__(self,
                                 name=name,
                                 tau=tau,
                                 gamma=None,
                                 config=config)


class TopicSelectionThetaRegularizer(BaseRegularizerTheta):
    _config_message = messages.TopicSelectionThetaConfig
    _type = const.RegularizerType_TopicSelectionTheta

    def __init__(self, name=None, tau=1.0, topic_names=None,
                 alpha_iter=None, config=None):
        """
        :param str name: the identifier of regularizer, will be auto-generated if not specified
        :param float tau: the coefficient of regularization for this regularizer
        :param alpha_iter: list of additional coefficients of regularization on each iteration\
                           over document. Should have length equal to model.num_document_passes
        :type alpha_iter: list of str
        :param topic_names: list of names of topics to regularize,\
                                     will regularize all topics if not specified
        :type topic_names: list of str
        :param config: the low-level config of this regularizer
        :type config: protobuf object
        """
        BaseRegularizerTheta.__init__(self,
                                      name=name,
                                      tau=tau,
                                      config=config,
                                      topic_names=topic_names,
                                      alpha_iter=alpha_iter)


class BitermsPhiRegularizer(BaseRegularizerPhi):
    _config_message = messages.BitermsPhiConfig
    _type = const.RegularizerType_BitermsPhi

    def __init__(self, name=None, tau=1.0, gamma=None, class_ids=None,
                 topic_names=None, dictionary=None, config=None):
        """
        :param str name: the identifier of regularizer, will be auto-generated if not specified
        :param float tau: the coefficient of regularization for this regularizer
        :param float gamma: the coefficient of relative regularization for this regularizer
        :param class_ids: list of class_ids to regularize, will\
                                     regularize all classes if not specified
        :type class_ids: list of str
        :param topic_names: list of names of topics to regularize,\
                                     will regularize all topics if not specified
        :type topic_names: list of str
        :param dictionary: BigARTM collection dictionary, won't use dictionary if not\
                                     specified, in this case regularizer is useless,
                                     dictionaty should contain pairwise tokens coocurancy info
        :type dictionary: str or reference to Dictionary object
        :param config: the low-level config of this regularizer
        :type config: protobuf object
        """
        BaseRegularizerPhi.__init__(self,
                                    name=name,
                                    tau=tau,
                                    gamma=gamma,
                                    config=config,
                                    topic_names=topic_names,
                                    class_ids=class_ids,
                                    dictionary=dictionary)
									
									
class HierarchySparsingThetaRegularizer(BaseRegularizerTheta):
    _config_message = messages.HierarchySparsingThetaConfig
    _type = const.RegularizerType_HierarchySparsingTheta

    def __init__(self, name=None, tau=1.0, topic_names=None,
                 alpha_iter=None, config=None):
        """
        :param str name: the identifier of regularizer, will be auto-generated if not specified
        :param float tau: the coefficient of regularization for this regularizer
        :param alpha_iter: list of additional coefficients of regularization on each iteration\
                           over document. Should have length equal to model.num_document_passes
        :type alpha_iter: list of str
        :param topic_names: list of names of topics to regularize,\
                                     will regularize all topics if not specified
        :type topic_names: list of str
        :param config: the low-level config of this regularizer
        :type config: protobuf object
        """
        BaseRegularizerTheta.__init__(self,
                                      name=name,
                                      tau=tau,
                                      config=config,
                                      topic_names=topic_names,
                                      alpha_iter=alpha_iter)
