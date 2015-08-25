import uuid

import artm.wrapper
import artm.wrapper.messages_pb2 as messages
import artm.wrapper.constants as constants


GLOB_EPS = 1e-37

__all__ = [
    'SparsityPhiScore',
    'ItemsProcessedScore',
    'PerplexityScore',
    'SparsityThetaScore',
    'ThetaSnippetScore',
    'TopicKernelScore',
    'TopTokensScore'
]

CREATE_SCORE_CONFIG = {
    constants.ScoreConfig_Type_SparsityPhi: messages.SparsityPhiScoreConfig,
    constants.ScoreConfig_Type_SparsityTheta: messages.SparsityThetaScoreConfig,
    constants.ScoreConfig_Type_Perplexity: messages.PerplexityScoreConfig,
    constants.ScoreConfig_Type_ItemsProcessed: messages.ItemsProcessedScoreConfig,
    constants.ScoreConfig_Type_TopTokens: messages.TopTokensScoreConfig,
    constants.ScoreConfig_Type_ThetaSnippet: messages.ThetaSnippetScoreConfig,
    constants.ScoreConfig_Type_TopicKernel: messages.TopicKernelScoreConfig
}


def _reconfigure_field(obj, field, field_name, proto_field_name=None):
    if proto_field_name is None:
        proto_field_name = field_name
    setattr(obj, '_' + field_name, field)
    score_config = CREATE_SCORE_CONFIG[obj._type]()
    score_config.CopyFrom(obj._config)
    if isinstance(field, list):
        score_config.ClearField(proto_field_name)
        for value in field:
            getattr(score_config, proto_field_name).append(value)
    else:
        setattr(score_config, proto_field_name, field)

    master_config = messages.MasterComponentConfig()
    master_config.CopyFrom(obj._master.config())
    for i in xrange(len(master_config.score_config)):
        if master_config.score_config[i].name == obj._name:
            master_config.score_config[i].config = score_config.SerializeToString()
            break
    obj._master.reconfigure(master_config)


class Scores(object):
    """Scores represents a storage of scores in ArtmModel (private class)

    Args:
      master (reference): reference to MasterComponent object, no default
    """

    def __init__(self, master, model):
        self._data = {}
        self._master = master
        self._model = model

    def add(self, score):
        """Scores.add() --- add score into ArtmModel.

        Args:
          score: an object of ***Scores class, no default
        """
        if score.name in self._data:
            raise ValueError('Score with name ' + str(score.name) + ' is already exist')
        else:
            self._master.create_score(score.name, score.type, score.config)
            score._model = self._model
            score._master = self._master
            self._data[score.name] = score

    def __getitem__(self, name):
        """Scores.__getitem__() --- get score with given name

        Args:
          name (str): name of the score, no default
        """
        if name in self._data:
            return self._data[name]
        else:
            raise KeyError('No score with name ' + name)

    @property
    def data(self):
        return self._data


class BaseScore(object):
    def __init__(self, name, class_id, topic_names):
        config = CREATE_SCORE_CONFIG[self._type]()
        self._class_id = '@default_class'
        self._topic_names = []

        if name is None:
            name = str(self._type) + uuid.uuid1().urn
        if class_id is not None:
            config.class_id = class_id
            self._class_id = class_id
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
                self._topic_names.append(topic_name)

        self._name = name
        self._config = config
        self._model = None  # Reserve place for the model
        self._master = None  # Reserve place for the master (to reconfigure Scores)

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
        _reconfigure_field(self, class_id, 'class_id')

    @topic_names.setter
    def topic_names(self, topic_names):
        _reconfigure_field(self, topic_names, 'topic_names')


###################################################################################################
# SECTION OF SCORE CLASSES
###################################################################################################
class SparsityPhiScore(BaseScore):
    """SparsityPhiScore is a score in ArtmModel (public class)

    Args:
      name (str): the identifier of score, will be auto-generated if not specified
      class_id (str): class_id to score, default=None
      topic_names (list of str): list of names of topics to regularize, will
      score all topics if not specified
      eps (double): the tolerance const, everything < eps
      considered to be zero, default=1e-37
    """

    _type = constants.ScoreConfig_Type_SparsityPhi

    def __init__(self, name=None, class_id=None, topic_names=None, eps=None):
        BaseScore.__init__(self,
                           name=name,
                           class_id=class_id,
                           topic_names=topic_names)
        self._eps = GLOB_EPS

        if eps is not None:
            self._config.eps = eps
            self._eps = eps

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, eps):
        _reconfigure_field(self, eps, 'eps')


class SparsityThetaScore(BaseScore):
    """SparsityThetaScore is a score in ArtmModel (public class)

    Args:
      name (str): the identifier of score, will be auto-generated if not specified
      topic_names (list of str): list of names of topics to regularize, will
      score all topics if not specified
      eps (double): the tolerance const, everything < eps
      considered to be zero, default=1e-37
    """

    _type = constants.ScoreConfig_Type_SparsityTheta

    def __init__(self, name=None, topic_names=None, eps=None):
        BaseScore.__init__(self,
                           name=name,
                           class_id=None,
                           topic_names=topic_names)
        self._eps = GLOB_EPS

        if eps is not None:
            self._config.eps = eps
            self._eps = eps

    @property
    def eps(self):
        return self._eps

    @property
    def class_id(self):
        raise KeyError('No class_id parameter')

    @eps.setter
    def eps(self, eps):
        _reconfigure_field(self, eps, 'eps')

    @class_id.setter
    def class_id(self, class_id):
        raise KeyError('No class_id parameter')


class PerplexityScore(BaseScore):
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

    _type = constants.ScoreConfig_Type_Perplexity

    def __init__(self, name=None, class_id=None, topic_names=None, eps=None,
                 dictionary_name=None, use_unigram_document_model=None):
        BaseScore.__init__(self,
                           name=name,
                           class_id=class_id,
                           topic_names=topic_names)
        self._eps = GLOB_EPS
        self._dictionary_name = ''
        self._use_unigram_document_model = True

        if eps is not None:
            self._config.theta_sparsity_eps = eps
            self._eps = eps
        if dictionary_name is not None:
            self._dictionary_name = dictionary_name
            self._config.dictionary_name = dictionary_name
        if use_unigram_document_model is not None:
            self._use_unigram_document_model = use_unigram_document_model
            if use_unigram_document_model is True:
                self._config.model_type = lib.PerplexityScoreConfig_Type_UnigramDocumentModel
            else:
                self._config.model_type = lib.PerplexityScoreConfig_Type_UnigramCollectionModel

    @property
    def eps(self):
        return self._eps

    @property
    def dictionary_name(self):
        return self._dictionary_name

    @property
    def use_unigram_document_model(self):
        return self._use_unigram_document_model

    @eps.setter
    def eps(self, eps):
        _reconfigure_field(self, eps, 'eps')

    @dictionary_name.setter
    def dictionary_name(self, dictionary_name):
        _reconfigure_field(self, dictionary_name, 'dictionary_name')

    @use_unigram_document_model.setter
    def use_unigram_document_model(self, use_unigram_document_model):
        self._use_unigram_document_model = use_unigram_document_model
        score_config = messages.PerplexityScoreConfig()
        score_config.CopyFrom(self._config)
        if use_unigram_document_model is True:
            score_config.model_type = lib.PerplexityScoreConfig_Type_UnigramDocumentModel
        else:
            score_config.model_type = lib.PerplexityScoreConfig_Type_UnigramCollectionModel
        _reconfigure_score_in_master(self._master, score_config, self._name)


class ItemsProcessedScore(BaseScore):
    """ItemsProcessedScore is a score in ArtmModel (public class)

    Args:
      name (str): the identifier of score, will be auto-generated if not specified
    """

    _type = constants.ScoreConfig_Type_ItemsProcessed

    def __init__(self, name=None):
        BaseScore.__init__(self,
                           name=name,
                           class_id=None,
                           topic_names=None)

    @property
    def topic_names(self):
        raise KeyError('No topic_names parameter')

    @property
    def class_id(self):
        raise KeyError('No class_id parameter')

    @topic_names.setter
    def topic_names(self, topic_names):
        raise KeyError('No topic_names parameter')

    @class_id.setter
    def class_id(self, class_id):
        raise KeyError('No class_id parameter')


class TopTokensScore(BaseScore):
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

    _type = constants.ScoreConfig_Type_TopTokens

    def __init__(self, name=None, class_id=None, topic_names=None,
                 num_tokens=None, dictionary_name=None):
        BaseScore.__init__(self,
                           name=name,
                           class_id=class_id,
                           topic_names=topic_names)
        self._num_tokens = 10
        self._dictionary_name = ''

        if num_tokens is not None:
            self._config.num_tokens = num_tokens
            self._num_tokens = num_tokens
        if dictionary_name is not None:
            self._dictionary_name = dictionary_name
            self._config.cooccurrence_dictionary_name = dictionary_name

    @property
    def num_tokens(self):
        return self._num_tokens

    @property
    def dictionary_name(self):
        return self._dictionary_name

    @num_tokens.setter
    def num_tokens(self, num_tokens):
        _reconfigure_field(self, num_tokens, 'num_tokens')

    @dictionary_name.setter
    def dictionary_name(self, dictionary_name):
        _reconfigure_field(self, dictionary_name,
                           'dictionary_name', 'cooccurrence_dictionary_name')


class ThetaSnippetScore(BaseScore):
    """ThetaSnippetScore is a score in ArtmModel (public class)

    Args:
      name (str): the identifier of score, will be auto-generated if not specified
      item_ids (list of int): list of names of items to show, default=None
      num_items (int): number of theta vectors to show from the
      beginning (no sense if item_ids given), default=10
    """

    _type = constants.ScoreConfig_Type_ThetaSnippet

    def __init__(self, name=None, item_ids=None, num_items=None):
        BaseScore.__init__(self,
                           name=name,
                           class_id=None,
                           topic_names=None)
        self._item_ids = []
        self._num_items = 10

        if item_ids is not None:
            self._config.ClearField('item_id')
            for item_id in item_ids:
                self._config.item_id.append(item_id)
                self._item_ids.append(item_id)
        if num_items is not None:
            self._config.item_count = num_items
            self._num_items = num_items

    @property
    def topic_names(self):
        raise KeyError('No topic_names parameter')

    @property
    def class_id(self):
        raise KeyError('No class_id parameter')

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
        _reconfigure_field(self, num_items, 'num_items', 'item_count')


class TopicKernelScore(BaseScore):
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

    _type = constants.ScoreConfig_Type_TopicKernel

    def __init__(self, name=None, class_id=None, topic_names=None, eps=None,
                 dictionary_name=None, probability_mass_threshold=None):
        BaseScore.__init__(self,
                           name=name,
                           class_id=class_id,
                           topic_names=topic_names)
        self._eps = GLOB_EPS
        self._dictionary_name = ''
        self._probability_mass_threshold = 0.1

        if eps is not None:
            self._config.theta_sparsity_eps = eps
            self._eps = eps
        if dictionary_name is not None:
            self._self._dictionary_name = dictionary_name
            config.cooccurrence_dictionary_name = dictionary_name
        if probability_mass_threshold is not None:
            self._config.probability_mass_threshold = probability_mass_threshold
            self._probability_mass_threshold = probability_mass_threshold

    @property
    def eps(self):
        return self._eps

    @property
    def dictionary_name(self):
        return self._dictionary_name

    @property
    def probability_mass_threshold(self):
        return self._probability_mass_threshold

    @eps.setter
    def eps(self, eps):
        _reconfigure_field(self, eps, 'eps')

    @dictionary_name.setter
    def dictionary_name(self, dictionary_name):
        _reconfigure_field(self, dictionary_name,
                           'dictionary_name', 'cooccurrence_dictionary_name')

    @probability_mass_threshold.setter
    def probability_mass_threshold(self, probability_mass_threshold):
        _reconfigure_field(self, probability_mass_threshold, 'probability_mass_threshold')
