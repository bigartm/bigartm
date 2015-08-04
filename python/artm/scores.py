import uuid
import collections

from pandas import DataFrame

import artm.messages_pb2 as messages_pb2
import artm.library as library


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
