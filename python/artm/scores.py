# Copyright 2017, Additive Regularization of Topic Models.

import uuid
import warnings

from . import wrapper
from .wrapper import messages_pb2 as messages
from .wrapper import constants as const

from six import string_types

GLOB_EPS = 1e-37

__all__ = [
    'PerplexityScore',
    'SparsityThetaScore',
    'SparsityPhiScore',
    'ItemsProcessedScore',
    'TopTokensScore',
    'ThetaSnippetScore',
    'TopicKernelScore',
    'TopicMassPhiScore',
    'ClassPrecisionScore',
    'BackgroundTokensRatioScore'
]


def _reconfigure_field(obj, field, field_name, proto_field_name=None):
    if proto_field_name is None:
        proto_field_name = field_name
    setattr(obj, '_{0}'.format(field_name), field)

    if isinstance(field, list):
        obj._config.ClearField(proto_field_name)
        for value in field:
            getattr(obj._config, proto_field_name).append(value)
    else:
        setattr(obj._config, proto_field_name, field)

    obj._master.reconfigure_score(obj._name, obj._config)


class Scores(object):
    def __init__(self, master, model_pwt, model_nwt):
        self._data = {}
        self._master = master
        self._model_pwt = model_pwt
        self._model_nwt = model_nwt

    def add(self, score, overwrite=False):
        name = score.name
        if name in self._data and not overwrite:
            raise AttributeError("Unable to replace existing score.\
                                  If you really want to do it use overwrite=True argument")
        # next statement represents ternary operator
        register_func = (self._master.create_score if name not in self._data else
                         self._master.reconfigure_score)
        register_func(name, score.config, score._model_name)
        score._model_pwt = self._model_pwt
        score._model_nwt = self._model_nwt
        score._master = self._master
        self._data[name] = score

    def __getitem__(self, name):
        if name in self._data:
            return self._data[name]
        else:
            raise KeyError('No score with name {0}'.format(name))

    def __setitem__(self, name, score):
        # typical usecase:
        # regs[name] = SomeScore(name=None, arguments)
        # or
        # regs[name] = SomeScore(arguments)
        # reset name of score
        # hack to make name substitution: we directly use _name
        score._name = name
        self.add(score, overwrite=True)

    def size(self):
        warnings.warn(DeprecationWarning(
            "Function 'size' is deprecated and will be removed soon,\
 use built-in function 'len' instead"))
        return len(self._data.keys())

    def __len__(self):
        return len(self._data.keys())

    @property
    def data(self):
        return self._data

    def __repr__(self):
        return '[{0}]'.format(', '.join(self._data))


class BaseScore(object):
    _config_message = None

    def __init__(self, name, class_id, topic_names, model_name, config):
        if self._config_message is None:
            raise NotImplementedError()
        self._config = config if config is not None else self._config_message()

        self._class_id = '@default_class'
        if class_id is not None:
            self._config.class_id = class_id
            self._class_id = class_id
        elif config is not None:
            try:
                self._class_id = config.class_id
            except AttributeError:
                pass

        self._topic_names = []
        if topic_names is not None:
            self._config.ClearField('topic_name')
            if isinstance(topic_names, string_types):
                topic_names = [topic_names]
            for topic_name in topic_names:
                self._config.topic_name.append(topic_name)
                self._topic_names.append(topic_name)
        elif config is not None:
            try:
                if len(config.topic_name):
                    self._topic_names = [topic_name for topic_name in config.topic_name]
            except AttributeError:
                pass

        self._name = name if name is not None else '{0}:{1}'.format(self._type, uuid.uuid1().urn)
        self._model_name = model_name if model_name is not None else 'pwt'
        self._model_pwt = None  # Reserve place for the model
        self._model_nwt = None  # Reserve place for the model
        self._master = None  # Reserve place for the master (to reconfigure Scores)

    @property
    def name(self):
        return self._name

    @property
    def config(self):
        return self._config

    @property
    def model_name(self):
        return self._model_name

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
    def model_pwt(self):
        return self._model_pwt

    @property
    def model_nwt(self):
        return self._model_nwt

    @property
    def score(self):
        return self._score

    @property
    def master(self):
        return self._master

    @class_id.setter
    def class_id(self, class_id):
        _reconfigure_field(self, class_id, 'class_id')

    @topic_names.setter
    def topic_names(self, topic_names):
        _reconfigure_field(self, topic_names, 'topic_names')

    @name.setter
    def name(self, name):
        raise RuntimeError("It's impossible to change score name")

    @model_name.setter
    def model_name(self, model_name):
        raise RuntimeError("It's impossible to change score model_name")


###################################################################################################
# SECTION OF SCORE CLASSES
###################################################################################################
class SparsityPhiScore(BaseScore):
    _config_message = messages.SparsityPhiScoreConfig
    _type = const.ScoreType_SparsityPhi

    def __init__(self, name=None, class_id=None, topic_names=None, model_name=None, eps=None, config=None):
        """
        :param str name: the identifier of score, will be auto-generated if not specified
        :param str class_id: class_id to score
        :param topic_names: list of names or single name of topic to regularize, will\
                            score all topics if empty or None
        :type topic_names: list of str or str or None
        :param model_name: phi-like matrix to be scored (typically 'pwt' or 'nwt'), 'pwt'\
                           if not specified
        :param float eps: the tolerance const, everything < eps considered to be zero
        :param config: the low-level config of this score
        :type config: protobuf object
        """
        BaseScore.__init__(self,
                           name=name,
                           class_id=class_id,
                           topic_names=topic_names,
                           model_name=model_name,
                           config=config)

        self._eps = GLOB_EPS
        if eps is not None:
            self._config.eps = eps
            self._eps = eps
        elif config is not None and config.HasField('eps'):
            self._eps = config.eps

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, eps):
        _reconfigure_field(self, eps, 'eps')


class SparsityThetaScore(BaseScore):
    _config_message = messages.SparsityThetaScoreConfig
    _type = const.ScoreType_SparsityTheta

    def __init__(self, name=None, topic_names=None, eps=None, config=None):
        """
        :param str name: the identifier of score, will be auto-generated if not specified
        :param topic_names: list of names or single name of topic to regularize, will\
                            score all topics if empty or None
        :type topic_names: list of str or str or None
        :param float eps: the tolerance const, everything < eps considered to be zero
        :param config: the low-level config of this score
        :type config: protobuf object
        """
        BaseScore.__init__(self,
                           name=name,
                           class_id=None,
                           topic_names=topic_names,
                           model_name=None,
                           config=config)

        self._eps = GLOB_EPS
        if eps is not None:
            self._config.eps = eps
            self._eps = eps
        elif config is not None and config.HasField('eps'):
            self._eps = config.eps

    @property
    def eps(self):
        return self._eps

    @property
    def class_id(self):
        raise KeyError('No class_id parameter')

    @property
    def model_name(self):
        raise KeyError('No model_name parameter')

    @eps.setter
    def eps(self, eps):
        _reconfigure_field(self, eps, 'eps')

    @class_id.setter
    def class_id(self, class_id):
        raise KeyError('No class_id parameter')

    @model_name.setter
    def model_name(self, model_name):
        raise KeyError('No model_name parameter')


class PerplexityScore(BaseScore):
    _config_message = messages.PerplexityScoreConfig
    _type = const.ScoreType_Perplexity

    def __init__(self, name=None, class_ids=None, dictionary=None, config=None):
        """
        :param str name: the identifier of score, will be auto-generated if not specified
        :param class_ids: class_id to score, means that tokens of all class_ids will be used
        :type class_ids: list of str
        :param dictionary: BigARTM collection dictionary, is strongly recommended to\
                           be used for correct replacing of zero counters.
        :type dictionary: str or reference to Dictionary object
        :param config: the low-level config of this score
        :type config: protobuf object
        """
        BaseScore.__init__(self,
                           name=name,
                           class_id=None,
                           topic_names=None,
                           model_name=None,
                           config=config)

        self._class_ids = []
        if class_ids is not None:
            self._config.ClearField('class_id')
            for class_id in class_ids:
                self._config.class_id.append(class_id)
                self._class_ids.append(class_id)
        elif config is not None and len(config.class_id):
            self._class_ids = [class_id for class_id in config.class_id]

        self._dictionary_name = ''
        if dictionary is not None:
            dictionary_name = dictionary if isinstance(dictionary, str) else dictionary.name
            self._dictionary_name = dictionary_name
            self._config.dictionary_name = dictionary_name
            self._config.model_type = const.PerplexityScoreConfig_Type_UnigramCollectionModel
        elif config is not None and config.HasField('dictionary_name'):
            self._dictionary_name = config.dictionary_name
        else:
            self._config.model_type = const.PerplexityScoreConfig_Type_UnigramDocumentModel

    @property
    def dictionary(self):
        return self._dictionary_name

    @property
    def class_ids(self):
        return self._class_ids

    @property
    def class_id(self):
        raise KeyError('No class_id parameter')

    @property
    def model_name(self):
        raise KeyError('No model_name parameter')

    @property
    def topic_names(self):
        raise KeyError('No topic_names parameter')

    @dictionary.setter
    def dictionary(self, dictionary):
        if len(self._dictionary_name) == 0:
            score_config = messages.PerplexityScoreConfig()
            score_config.CopyFrom(self._config)
            score_config.model_type = const.PerplexityScoreConfig_Type_UnigramCollectionModel
            self._master.reconfigure_score(self._name, score_config)

        dictionary_name = dictionary if isinstance(dictionary, str) else dictionary.name
        _reconfigure_field(self, dictionary_name, 'dictionary_name')

    @class_ids.setter
    def class_ids(self, class_ids):
        _reconfigure_field(self, class_ids, 'class_ids', 'class_id')

    @class_id.setter
    def class_id(self, class_id):
        raise KeyError('No class_id parameter')

    @model_name.setter
    def model_name(self, model_name):
        raise KeyError('No model_name parameter')

    @topic_names.setter
    def topic_names(self, topic_names):
        raise KeyError('No topic_names parameter')


class ItemsProcessedScore(BaseScore):
    _config_message = messages.ItemsProcessedScoreConfig
    _type = const.ScoreType_ItemsProcessed

    def __init__(self, name=None, config=None):
        """
        :param str name: the identifier of score, will be auto-generated if not specified
        :param config: the low-level config of this score
        :type config: protobuf object
        """
        BaseScore.__init__(self,
                           name=name,
                           class_id=None,
                           topic_names=None,
                           model_name=None,
                           config=config)

    @property
    def topic_names(self):
        raise KeyError('No topic_names parameter')

    @property
    def class_id(self):
        raise KeyError('No class_id parameter')

    @property
    def model_name(self):
        raise KeyError('No model_name parameter')

    @topic_names.setter
    def topic_names(self, topic_names):
        raise KeyError('No topic_names parameter')

    @class_id.setter
    def class_id(self, class_id):
        raise KeyError('No class_id parameter')

    @model_name.setter
    def model_name(self, model_name):
        raise KeyError('No model_name parameter')


class TopTokensScore(BaseScore):
    _config_message = messages.TopTokensScoreConfig
    _type = const.ScoreType_TopTokens

    def __init__(self, name=None, class_id=None, topic_names=None,
                 num_tokens=None, dictionary=None, config=None):
        """
        :param str name: the identifier of score, will be auto-generated if not specified
        :param str class_id: class_id to score
        :param topic_names: list of names or single name of topic to regularize, will\
                            score all topics if empty or None
        :type topic_names: list of str or str or None
        :param int num_tokens: number of tokens with max probability in each topic
        :param dictionary: BigARTM collection dictionary, won't use\
                            dictionary if not specified
        :type dictionary: str or reference to Dictionary object
        :param config: the low-level config of this score
        :type config: protobuf object
        """
        BaseScore.__init__(self,
                           name=name,
                           class_id=class_id,
                           topic_names=topic_names,
                           model_name=None,
                           config=config)

        self._num_tokens = 10
        if num_tokens is not None:
            self._config.num_tokens = num_tokens
            self._num_tokens = num_tokens
        elif config is not None and config.HasField('num_tokens'):
            self._num_tokens = config.num_tokens

        self._dictionary_name = ''
        if dictionary is not None:
            dictionary_name = dictionary if isinstance(dictionary, str) else dictionary.name
            self._dictionary_name = dictionary_name
            self._config.cooccurrence_dictionary_name = dictionary_name
        elif config is not None and config.HasField('cooccurrence_dictionary_name'):
            self._dictionary_name = config.cooccurrence_dictionary_name

    @property
    def num_tokens(self):
        return self._num_tokens

    @property
    def dictionary(self):
        return self._dictionary_name

    @property
    def model_name(self):
        raise KeyError('No model_name parameter')

    @num_tokens.setter
    def num_tokens(self, num_tokens):
        _reconfigure_field(self, num_tokens, 'num_tokens')

    @dictionary.setter
    def dictionary(self, dictionary):
        dictionary_name = dictionary if isinstance(dictionary, str) else dictionary.name
        _reconfigure_field(self, dictionary_name,
                           'dictionary_name', 'cooccurrence_dictionary_name')

    @model_name.setter
    def model_name(self, model_name):
        raise KeyError('No model_name parameter')


class ThetaSnippetScore(BaseScore):
    _config_message = messages.ThetaSnippetScoreConfig
    _type = const.ScoreType_ThetaSnippet

    def __init__(self, name=None, item_ids=None, num_items=None, config=None):
        """
        :param str name: the identifier of score, will be auto-generated if not specified
        :param item_ids: list of names of items to show, default=None
        :type item_ids: list of int
        :param int num_items: number of theta vectors to show from the beginning\
                                (no sense if item_ids was given)
        :param config: the low-level config of this score
        :type config: protobuf object
        """
        BaseScore.__init__(self,
                           name=name,
                           class_id=None,
                           topic_names=None,
                           model_name=None,
                           config=config)

        self._item_ids = []
        if item_ids is not None:
            self._config.ClearField('item_id')
            for item_id in item_ids:
                self._config.item_id.append(item_id)
                self._item_ids.append(item_id)
        elif config is not None and len(config.item_id):
            self._item_ids = [item_id for item_id in config.item_id]

        self._num_items = 10
        if num_items is not None:
            self._config.num_items = num_items
            self._num_items = num_items
        elif config is not None and config.HasField('num_items'):
            self._num_items = config.num_items

    @property
    def topic_names(self):
        raise KeyError('No topic_names parameter')

    @property
    def class_id(self):
        raise KeyError('No class_id parameter')

    @property
    def model_name(self):
        raise KeyError('No model_name parameter')

    @property
    def item_ids(self):
        return self._item_ids

    @property
    def num_items(self):
        return self._num_items

    @topic_names.setter
    def topic_names(self, topic_names):
        raise KeyError('No topic_names parameter')

    @class_id.setter
    def class_id(self, class_id):
        raise KeyError('No class_id parameter')

    @item_ids.setter
    def item_ids(self, item_ids):
        _reconfigure_field(self, item_ids, 'item_ids', 'item_id')

    @num_items.setter
    def num_items(self, num_items):
        _reconfigure_field(self, num_items, 'num_items', 'num_items')

    @model_name.setter
    def model_name(self, model_name):
        raise KeyError('No model_name parameter')


class TopicKernelScore(BaseScore):
    _config_message = messages.TopicKernelScoreConfig
    _type = const.ScoreType_TopicKernel

    def __init__(self, name=None, class_id=None, topic_names=None, eps=None,
                 dictionary=None, probability_mass_threshold=None, config=None):
        """
        :param str name: the identifier of score, will be auto-generated if not specified
        :param str class_id: class_id to score
        :param topic_names: list of names or single name of topic to regularize, will\
                            score all topics if empty or None
        :type topic_names: list of str or str or None
        :param float probability_mass_threshold: the threshold for p(t|w) values to get\
                            token into topic kernel. Should be in (0, 1)
        :param dictionary: BigARTM collection dictionary, won't use\
                            dictionary if not specified
        :type dictionary: str or reference to Dictionary object
        :param float eps: the tolerance const, everything < eps considered to be zero
        :param config: the low-level config of this score
        :type config: protobuf object
        """
        BaseScore.__init__(self,
                           name=name,
                           class_id=class_id,
                           topic_names=topic_names,
                           model_name=None,
                           config=config)

        self._eps = GLOB_EPS
        if eps is not None:
            self._config.eps = eps
            self._eps = eps
        elif config is not None and config.HasField('eps'):
            self._eps = config.eps

        self._dictionary_name = ''
        if dictionary is not None:
            dictionary_name = dictionary if isinstance(dictionary, str) else dictionary.name
            self._dictionary_name = dictionary_name
            self._config.cooccurrence_dictionary_name = dictionary_name
        elif config is not None and config.HasField('cooccurrence_dictionary_name'):
            self._dictionary_name = config.cooccurrence_dictionary_name

        self._probability_mass_threshold = 0.1
        if probability_mass_threshold is not None:
            self._config.probability_mass_threshold = probability_mass_threshold
            self._probability_mass_threshold = probability_mass_threshold
        elif config is not None and config.HasField('probability_mass_threshold'):
            self._probability_mass_threshold = config.probability_mass_threshold

    @property
    def eps(self):
        return self._eps

    @property
    def dictionary(self):
        return self._dictionary_name

    @property
    def probability_mass_threshold(self):
        return self._probability_mass_threshold

    @property
    def model_name(self):
        raise KeyError('No model_name parameter')

    @eps.setter
    def eps(self, eps):
        _reconfigure_field(self, eps, 'eps')

    @dictionary.setter
    def dictionary(self, dictionary):
        dictionary_name = dictionary if isinstance(dictionary, str) else dictionary.name
        _reconfigure_field(self, dictionary_name,
                           'dictionary_name', 'cooccurrence_dictionary_name')

    @probability_mass_threshold.setter
    def probability_mass_threshold(self, probability_mass_threshold):
        _reconfigure_field(self, probability_mass_threshold, 'probability_mass_threshold')

    @model_name.setter
    def model_name(self, model_name):
        raise KeyError('No model_name parameter')


class TopicMassPhiScore(BaseScore):
    _config_message = messages.TopicMassPhiScoreConfig
    _type = const.ScoreType_TopicMassPhi

    def __init__(self, name=None, class_ids=None, topic_names=None, model_name=None, eps=None, config=None):
        """
        :param str name: the identifier of score, will be auto-generated if not specified
        :param class_ids: class_id to score, means that tokens of all class_ids will be used
        :type class_ids: list of str
        :param topic_names: list of names or single name of topic to regularize, will\
                            score all topics if empty or None
        :type topic_names: list of str or str or None
        :param model_name: phi-like matrix to be scored (typically 'pwt' or 'nwt'), 'pwt'\
                           if not specified
        :param float eps: the tolerance const, everything < eps considered to be zero
        :param config: the low-level config of this score
        :type config: protobuf object
        """
        BaseScore.__init__(self,
                           name=name,
                           class_id=None,
                           topic_names=topic_names,
                           model_name=model_name,
                           config=config)

        self._eps = GLOB_EPS
        if eps is not None:
            self._config.eps = eps
            self._eps = eps
        elif config is not None and config.HasField('eps'):
            self._eps = config.eps

        self._class_ids = []
        if class_ids is not None:
            self._config.ClearField('class_id')
            for class_id in class_ids:
                self._config.class_id.append(class_id)
                self._class_ids.append(class_id)
        elif config is not None and len(config.class_id):
            self._class_ids = [class_id for class_id in config.class_id]

    @property
    def eps(self):
        return self._eps

    @property
    def class_ids(self):
        return self._class_ids

    @property
    def class_id(self):
        raise KeyError('No class_id parameter')

    @eps.setter
    def eps(self, eps):
        _reconfigure_field(self, eps, 'eps')

    @class_ids.setter
    def class_ids(self, class_ids):
        _reconfigure_field(self, class_ids, 'class_ids', 'class_id')

    @class_id.setter
    def class_id(self, class_id):
        raise KeyError('No class_id parameter')


class ClassPrecisionScore(BaseScore):
    _config_message = messages.ClassPrecisionScoreConfig
    _type = const.ScoreType_ClassPrecision

    def __init__(self, name=None, config=None):
        """
        :param str name: the identifier of score, will be auto-generated if not specified
        :param config: the low-level config of this score
        :type config: protobuf object
        """
        BaseScore.__init__(self,
                           name=name,
                           class_id=None,
                           topic_names=None,
                           model_name=None,
                           config=config)

    @property
    def topic_names(self):
        raise KeyError('No topic_names parameter')

    @property
    def class_id(self):
        raise KeyError('No class_id parameter')

    @property
    def model_name(self):
        raise KeyError('No model_name parameter')

    @topic_names.setter
    def topic_names(self, topic_names):
        raise KeyError('No topic_names parameter')

    @class_id.setter
    def class_id(self, class_id):
        raise KeyError('No class_id parameter')

    @model_name.setter
    def model_name(self, model_name):
        raise KeyError('No model_name parameter')


class BackgroundTokensRatioScore(BaseScore):
    _config_message = messages.BackgroundTokensRatioScoreConfig
    _type = const.ScoreType_BackgroundTokensRatio

    def __init__(self, name=None, class_id=None, delta_threshold=None,
                 save_tokens=None, direct_kl=None, config=None):
        """
        :param str name: the identifier of score, will be auto-generated if not specified
        :param str class_id: class_id to score
        :param float delta_threshold: the threshold for KL-div between p(t|w) and p(t) to get\
                            token into background. Should be non-negative
        :param bool save_tokens: save background tokens or not, save if field not specified
        :param bool direct_kl: use KL(p(t) || p(t|w)) or via versa, true if field not specified
        :param config: the low-level config of this score
        :type config: protobuf object
        """
        BaseScore.__init__(self,
                           name=name,
                           class_id=class_id,
                           topic_names=None,
                           model_name=None,
                           config=config)

        self._save_tokens = True
        if save_tokens is not None:
            self._config.save_tokens = save_tokens
            self._save_tokens = save_tokens
        elif config is not None and config.HasField('save_tokens'):
            self._save_tokens = config.save_tokens

        self._direct_kl = True
        if direct_kl is not None:
            self._config.direct_kl = direct_kl
            self._direct_kl = direct_kl
        elif config is not None and config.HasField('direct_kl'):
            self._direct_kl = config.direct_kl

        self._delta_threshold = 0.5
        if delta_threshold is not None:
            self._config.delta_threshold = delta_threshold
            self._delta_threshold = delta_threshold
        elif config is not None and config.HasField('delta_threshold'):
            self._delta_threshold = config.delta_threshold

    @property
    def save_tokens(self):
        return self._save_tokens

    @property
    def direct_kl(self):
        return self._direct_kl

    @property
    def delta_threshold(self):
        return self._delta_threshold

    @property
    def model_name(self):
        raise KeyError('No model_name parameter')

    @property
    def topic_names(self):
        raise KeyError('No topic_names parameter')

    @save_tokens.setter
    def save_tokens(self, save_tokens):
        _reconfigure_field(self, save_tokens, 'save_tokens')

    @direct_kl.setter
    def direct_kl(self, direct_kl):
        _reconfigure_field(self, direct_kl, 'direct_kl')

    @delta_threshold.setter
    def delta_threshold(self, delta_threshold):
        _reconfigure_field(self, delta_threshold, 'delta_threshold')

    @model_name.setter
    def model_name(self, model_name):
        raise KeyError('No model_name parameter')

    @topic_names.setter
    def topic_names(self, topic_names):
        raise KeyError('No topic_names parameter')
