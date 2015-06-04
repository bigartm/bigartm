import artm.messages_pb2, artm.library, random, uuid, sys, glob, numpy, random
from numpy import matrix

# forward declaration
class Regularizers(object): pass
class Scores(object): pass

class SmoothSparsePhiRegularizer(object): pass
class SmoothSparseThetaRegularizer(object): pass
class DecorrelatorPhiRegularizer(object): pass
class LabelRegularizationPhiRegularizer(object): pass
class SpecifiedSparsePhiRegularizer(object): pass
class ImproveCoherencePhiRegularizer(object): pass

class SparsityPhiScore(object): pass
class SparsityThetaScore(object): pass
class PerplexityScore(object): pass
class ItemsProcessedScore(object): pass
class TopTokensScore(object): pass
class ThetaSnippetPhiScore(object): pass
class TopicKernelPhiScore(object): pass

class SparsityPhiScoreInfo(object): pass
class SparsityThetaScoreInfo(object): pass
class PerplexityScoreInfo(object): pass
class ItemsProcessedScoreInfo(object): pass
class TopTokensScoreInfo(object): pass
class ThetaSnippetPhiScoreInfo(object): pass
class TopicKernelPhiScoreInfo(object): pass

#######################################################################################################################
class ArtmModel(object):
  """ ArtmModel represents a topic model (public class).
  Args:
  - num_processors --- how many threads will be used for model training. Is int, default = 1
  - topic_names --- names of topics in model. Is list of strings, default = []
  - topics_count --- number of topics in model (is used if topic_names == []). Is int, default = 10
  - class_ids --- list of class_ids and their weights to be used in model. Is dict, 
                  key --- class_id, value --- weight, default = {}
  - inner_iterations_count --- number of iterations over each document during processing/ Is int, default = 1

  Important public fields:
  - regularizers --- contains dict of regularizers, included into model
  - scores --- contains dict of scores, included into model
  - scores_info --- contains dict of scoring results; key --- score name, value --- ScoreInfo object, which 
                    contains info about values of score on each syncronization in list
  NOTE:
  Here and anywhere in BigARTM empty topic_names or class_ids means that model (or regularizer, or score) should 
  use all topics or class_ids. If some fields of regularizers or scores are not defined by user --- internal library 
  defaults would be used.
  """

######### CONSTRUCTOR #########
  def __init__(self, num_processors=1, topic_names=[], topics_count=10, class_ids={}, inner_iterations_count=1):
    self._num_processors = 1
    self._topics_count = 10
    self._topic_names = []
    self._class_ids = {}
    self._inner_iterations_count = 1

    if num_processors > 0:         self._num_processors         = num_processors
    if topics_count > 0:           self._topics_count           = topics_count
    if len(topic_names) > 0:       self._topic_names            = topic_names
    if len(class_ids) > 0:         self._class_ids              = class_ids
    if inner_iterations_count > 0: self._inner_iterations_count = inner_iterations_count

    self._master = artm.library.MasterComponent()
    self._master.config().processors_count = self._num_processors
    self._master.Reconfigure()     

    model_config = artm.messages_pb2.ModelConfig()
    model_config.name = 'Model'
    if len(self._topic_names) > 0:
      for topic_name in self._topic_names:
        model_config.topic_name.append(topic_name)
      self._topics_count = len(self._topic_names)
    else:
      model_config.topics_count = self._topics_count
        
    if len(self._class_ids) > 0:
      for class_id, class_weight in self._class_ids:
        model_config.class_id.append(class_id)
        model_config.class_weight.append(class_weight)
    model_config.inner_iterations_count = self._inner_iterations_count
    self._model = self._master.CreateModel(config=model_config)

    self._regularizers = Regularizers(self._master, self._model)
    self._scores = Scores(self._master, self._model)

    self._scores_info = {}
    self._syncronizations_processed = -1 

######### PROPERTIES #########
  @property
  def num_processors(self): return self._num_processors
  @property
  def inner_iterations_count(self): return self._inner_iterations_count
  @property
  def tokens_count(self): return self._tokens_count
  @property
  def topics_count(self): return self._topics_count
  @property
  def topic_names(self): return self._topic_names
  @property
  def class_ids(self): return self._class_ids
  @property
  def regularizers(self): return self._regularizers
  @property
  def scores(self): return self._scores
  @property
  def scores_info(self): return self._scores_info

######### SETTERS #########
  @num_processors.setter
  def num_processors(self, num_processors):
    if num_processors <= 0 or not isinstance(num_processors, int):
      print 'Number of processors should be a positive integer, skip update'
    else:
      self._num_processors = num_processors
      self._master.config().processors_count = num_processors
      self._master.Reconfigure()
      
  @inner_iterations_count.setter
  def inner_iterations_count(self, inner_iterations_count):
    if inner_iterations_count <= 0 or not isinstance(inner_iterations_count, int):
      print 'Number of inner iterations should be a positive integer, skip update'
    else:
      self._inner_iterations_count = inner_iterations_count
      config = artm.messages_pb2.ModelConfig()
      config.CopyFrom(self._model.config())
      config.inner_iterations_count = inner_iterations_count
      self._model.Reconfigure(config)

  @topics_count.setter
  def topics_count(self, topics_count):
    if topics_count <= 0 or not isinstance(topics_count, int):
      print 'Number of topics should be a positive integer, skip update'
    else:
      self._topics_count = topics_count
      config = artm.messages_pb2.ModelConfig()
      config.CopyFrom(self._model.config())
      config.topics_count = topics_count
      self._model.Reconfigure(config)

  @topic_names.setter
  def topic_names(self, topic_names):
    if len(topic_names) < 0:
      print 'Number of topic names should be non-negative, skip update'
    else:
      self._topic_names = topic_names
      config = artm.messages_pb2.ModelConfig()
      config.CopyFrom(self._model.config())
      config.ClearField('topic_name')
      for topic_name in topic_names:
        config.topic_name.append(name)
      self._model.Reconfigure(config)

  @class_ids.setter
  def class_ids(self, class_ids):
    if len(class_ids) < 0:
      print 'Number of (class_id, class_weight) pairs shoul be non-negative, skip update'
    else:
      config = artm.messages_pb2.ModelConfig()
      config.CopyFrom(self._model.config())
      config.ClearField('class_id')
      for class_id in class_ids:
        config.class_id.append(class_id)

      config.ClearField('class_weight')
      for weight in class_weight:
        config.class_weight.weight()
      self._model.Reconfigure(config)

######### METHODS #########
  def fit(self, data_path='', outer_iterations_count=1, decay_weight=0.9,
          apply_weight=0.1, update_every=1, data_format='batches',
          batch_size=1000, gather_cooc=False, cooc_tokens=[]):
    """ ArtmModel.fit() --- proceed the learning of topic model
    Args:
    - data_path --- 1) if data_format == 'batches' => folder containing batches and dictionary
                    2) if data_format == 'bow_uci' => folder containig docword.collection_name.txt 
                                                      and vocab.collection_name.txt files
                    3) if data_format == 'bow_vw' => file in Vowpal Wabbit format
                    4) if data_format == 'plain_text' => file with text
                    Is string, default = ''
    - outer_iterations_count --- number of iterations over whole given collection. Is int, default = 1
    - decay_weight --- coefficient for applying old n_wt counters. Is int, default = 0.9
    - apply_weight --- coefficient for applying new n_wt counters. Is int, default = 0.1
    - update_every --- the number of batches; model will be updated once per it. Is int, default = 1
    - data_format --- the type of input data: 1) 'batches' --- the data in format of BigARTM
                                              2) 'bow_uci' --- Bag-Of-Words in UCI format
                                              3) 'bow_vw' --- Bag-Of-Words in Vowpal Wabbit format
                                              4) 'plain_text' --- source text
                                              Is string, default = 'batches'
    Next three arguments have sence only if data_format is not 'batches' (e.g. parsing is necessary).
    - batch_size --- number of documnets to be stored in each batch. Is int, default = 1000
    - gather_cooc --- find or not the info about the token pairwise co-occuracies. Is bool, default=False
    - cooc_tokens --- tokens to collect cooc info (has sence if gather_cooc is True). Is list of lists, each 
                      internal list represents token and contain two strings --- token and its class_id, default = []
    """

    unique_tokens = artm.messages_pb2.DictionaryConfig()
    if data_format == 'batches':
      batches = glob.glob(data_path + "/*.batch")
      if len(batches) < 1:
        print 'No batches were found, skip model.fit()'
        return
      print 'Found ' + str(len(batches)) + ' batches, using them.'
      unique_tokens = artm.library.Library().LoadDictionary(data_path + '/dictionary')

    elif data_format == 'bow_uci':
      collection_parser_config = artm.messages_pb2.CollectionParserConfig()
      collection_parser_config.format = artm.library.CollectionParserConfig_Format_BagOfWordsUci
      collection_parser_config.num_items_per_batch = batch_size

      collection_parser_config.docword_file_path = data_path + 'docword.' + collection_name + '.txt'
      collection_parser_config.vocab_file_path = data_path + 'vocab.' + collection_name + '.txt'
      collection_parser_config.target_folder = data_path
      collection_parser_config.dictionary_file_name = 'dictionary'
    
      collection_parser_config.gather_cooc = gather_cooc
      for token in cooc_tokens:
        collection_parser_config.cooccurrence_token.append(token)
      unique_tokens = artm.library.Library().ParseCollection(collection_parser_config)

    elif data_format == 'bow_vw':
      raise NotImplementedError()
    elif data_format == 'plain_text':
      raise NotImplementedError()
    else:
      print 'Unknown data format, skip model.fit()'

    for iter in range(outer_iterations_count):
      for batch_index, batch_filename in enumerate(batches):
        self._master.AddBatch(batch_filename=batch_filename)
        if ((batch_index + 1) % update_every == 0) or ((batch_index + 1) == len(batches)):
            self._master.WaitIdle()
            self._model.Synchronize(decay_weight=decay_weight, apply_weight=apply_weight)
            self._syncronizations_processed +=1

            for name in self.scores.data.keys():
              if not name in self.scores_info:
                if (self.scores[name].type == artm.library.ScoreConfig_Type_SparsityPhi):
                  self._scores_info[name] = SparsityPhiScoreInfo(self.scores[name])
                else:
                  raise NotImplementedError()

                for i in range(self._syncronizations_processed):
                  self._scores_info[name].add()
              self._scores_info[name].add(self.scores[name])

  
  def save(self, file_name='artm_model'):
    """ ArtmModel.save() --- save the topic model to disk.
    Args:
    - file_name --- the name of file to store model. Is string, default = 'artm_model'
    """
    with open(file_name, 'wb') as binary_file:
      binary_file.write(self._master.GetTopicModel(self._model).SerializeToString())

  def load(self, file_name):
    """ ArtmModel.load() --- load the topic model, saved by ArtmModel.save(), from disk.
    Args:
    - file_name --- the name of file containing model. Is string, no default
    """
    topic_model = artm.messages_pb2.TopicModel()
    with open(file_name, 'rb') as binary_file:
      topic_model.ParseFromString(binary_file.read())
    topic_model.ClearField('operation_type')
    for token in topic_model.token:
      topic_model.operation_type.append(artm.library.TopicModel_OperationType_Overwrite)
    self._model.Overwrite(topic_model=topic_model, commit=True)

    # remove all info about previous iterations
    self._scores_info = {}
    self._syncronizations_processed = -1
  
  def to_csv(self, file_name='artm_model.csv'):
    """ ArtmModel.to_csv() --- save the topic model to disk in .csv format (can't be loaded back).
    Args:
    - file_name --- the name of file to store model. Is string, default = 'artm_model.csv'
    """
    raise NotImplementedError()

  def get_theta(self):
    """ ArtmModel.get_theta() --- get Theta matrix for training set of documents.
    """
    raise NotImplementedError()
  
  def find_theta(self, data_path='', data_format='batches'):
    """ ArtmModel.find_theta() --- find Theta matrix for new documents.
    Args:
    - data_path --- 1) if data_format == 'batches' => folder containing batches and dictionary
                    2) if data_format == 'bow_uci' => folder containig docword.txt and vocab.txt files
                    3) if data_format == 'bow_vw' => file in Vowpal Wabbit format
                    4) if data_format == 'plain_text' => file with text
                    Is string, default = ''
    - data_format --- the type of input data: 1) 'batches' --- the data in format of BigARTM
                                              2) 'bow_uci' --- Bag-Of-Words in UCI format
                                              3) 'bow_vw' --- Bag-Of-Words in Vowpal Wabbit format
                                              4) 'plain_text' --- source text
                                              Is string, default = 'batches'
    """
    raise NotImplementedError()
  
  def init(self, np_matrix=None, tokens=None, dictionary=None):
    """ ArtmModel.init() --- initialize topic model before learning.
    Args:
    - np_matrix --- matrix, containing counters; size is len(tokens) x ArtmModel.topics_count.
                    Is numpy.matrix, default = None
    - tokens --- list of all tokens in vocabulary. Is list of lists, each internal list represents 
                 token and contain two strings --- token and its class_id, default = None
    - dictionary --- BigARTM collection dictionary. Is string, default = None

    Priority of initialization:
    1) tokens + np_atrix
    2) tokens + random().random()
    3) dictionary [analogue of 2)]
    """
    if not tokens is None:
      topic_model = artm.messages_pb2.TopicModel()
      topic_model.name = self._model.name
      topic_model.topics_count = self._topics_count
      for name in self._topic_names:
        topic_model.topic_name.append(name)
        token_id = -1
      for token in tokens:
        token_id += 1
        topic_model.operation_type.append(artm.library.TopicModel_OperationType_Overwrite)
        topic_model.token.append(token[0])
        topic_model.class_id.append(token[1])
        token_weights = topic_model.token_weights.add()
        for topic_id in range(0, topics_count):
          if not np_matrix is None: 
            token_weights.value.append(matrix[token_id, topic_id])
          else:
            token_weights.value.append(random.random())
      self._model.Overwrite(topic_model=topic_model, commit=True)
    elif not dictionary is None:
      self._model.Initialize(dictionary=dictionary)
    else:
      print 'Not enough arguments for initialization, skip it'
  
  def apply_regularization(self, additions=None):
    raise NotImplementedError()

#######################################################################################################################  
class Regularizers(object):
  """ Regularizers represents a storage of regularizers in ArtmModel (private class).
  Args:
  - master --- reference to master component object, no default
  - model --- reference to model object, no default
  """
  def __init__(self, master, model):
    self._data = {}
    self._master = master
    self._model = model

  def add(self, config):
    """ Regularizers.add() --- add regularizer into ArtmModel.
    Args:
    - config --- an object of ***Regularizer class, no default
    """
    if config.name in self._data:
      print 'Regularizer with name ' + str(config.name) + ' is already exist'
    else:
      regularizer = self._master.CreateRegularizer(config.name, config.type, config.config)
      config_copy = artm.messages_pb2.ModelConfig()
      config_copy.CopyFrom(self._model.config())
      settings = config_copy.regularizer_settings.add()
      settings.name = config.name
      settings.tau = config.tau
      self._model.Reconfigure(config_copy)

      config.model = self._model
      config.regularizer = regularizer
      self._data[config.name] = config
 
  def __getitem__(self, name):
    """ Regularizers.__getitem__() --- get regularizer with given name.
    Args:
    - name --- name of the regulrizer. Is string, no default
    """
    if name in self._data:
      return self._data[name]
    else:
      print 'No regularizer with name ' + str(config.name)
  
  @property
  def data(self): return self._data

#######################################################################################################################
class Scores(object):
  """ Scores represents a storage of scores in ArtmModel (private class).
  Args:
  - master --- reference to master component object, no default
  - model --- reference to model object, no default
  """
  def __init__(self, master, model):
    self._data = {}
    self._master = master
    self._model = model

  def add(self, config):
    """ Scores.add() --- add score into ArtmModel.
    Args:
    - config --- an object of ***Scores class, no default
    """
    if config.name in self._data:
      print 'Score with name ' + str(config.name) + ' is already exist'
    else:
      score = self._master.CreateScore(config.name, config.type, config.config)
      config_copy = artm.messages_pb2.ModelConfig()
      config_copy.CopyFrom(self._model.config())
      settings = config_copy.score_name.append(config.name)
      self._model.Reconfigure(config_copy)

      config.model = self._model
      config.score = score
      config.master = self._master
      self._data[config.name] = config
 
  def __getitem__(self, name):
    """ Scores.__getitem__() --- get score with given name.
    Args:
    - name --- name of the score. Is string, no default
    """
    if name in self._data:
      return self._data[name]
    else:
      print 'No score with name ' + str(config.name)

  @property
  def data(self): return self._data

#######################################################################################################################
# SECTION OF REGULARIZER CLASSES
#######################################################################################################################
class SmoothSparsePhiRegularizer(object):
  """ SmoothSparsePhiRegularizer is a regularizer in ArtmModel (public class).
  Args:
  - name --- the identificator of regularizer. Is string, default = None
  - tau --- the coefficient of regularization for this regularizer, double, default = 1.0
  - class_ids --- list of class_ids to regularize. Is list of strings, default = None
  - topic_names --- list of names of topics to regularize. Is list of strings, default = None
  - dictionary --- BigARTM collection dictionary. Is string, default = None
  """
  def __init__(self, name=None, tau=1.0, class_ids=None, topic_names=None, dictionary_name=None):
    config = artm.messages_pb2.SmoothSparsePhiConfig()
    self._class_ids = []
    self._topic_names = []
    self._dictionary_name = ''

    if name is None:
      name = "SmoothSparsePhiRegularizer:" + uuid.uuid1().urn
    if not class_ids is None:
      config.ClearField('class_id')
      for class_id in class_ids:
        config.class_id.append(class_id)
        self._class_ids.append(class_id)
    if not topic_names is None:
      config.ClearField('topic_name')
      for topic_name in topic_names:
        config.topic_name.append(topic_name)
        self._topic_names.append(topic_name)
    if not dictionary_name is None:
      config.dictionary_name = dictionary_name
      self._dictionary_name = dictionary_name

    self._name = name
    self._tau = tau
    self._config = config
    self._type = artm.library.RegularizerConfig_Type_SmoothSparsePhi
    self._regularizer = None  # reserve place for regularizer
    self._model = None  # reserve place for model
    
  @property
  def name(self): return self._name
  @property
  def tau(self): return self._tau 
  @property
  def config(self): return self._config  
  @property
  def type(self): return self._type
  @property
  def class_ids(self): return self._class_ids
  @property
  def topic_names(self): return self._topic_names
  @property
  def dictionary_name(self): return self._dictionary_name
  @property
  def model(self): return self._model
  @property
  def regularizer(self): return self._regularizer


  @model.setter
  def model(self, model): self._model = model
  @regularizer.setter
  def regularizer(self, regularizer): self._regularizer = regularizer

  @tau.setter
  def tau(self, tau):
    self._tau = tau
    config = artm.messages_pb2.ModelConfig()
    config.CopyFrom(self._model.config())
    for i in range(len(config.regularizer_settings)):
      if config.regularizer_settings[i].name == self._name:
        config.regularizer_settings[i].tau = tau
        break
    self._model.Reconfigure(config)

  @class_ids.setter
  def class_ids(self, class_ids):
    self._class_ids = class_ids
    config = artm.messages_pb2.SmoothSparsePhiConfig()
    config.CopyFrom(self._config)
    config.ClearField('class_id')
    for class_id in class_ids:
      config.class_id.append(class_id)
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

  @topic_names.setter
  def topic_names(self, topic_names):
    self._topic_names = topic_names
    config = artm.messages_pb2.SmoothSparsePhiConfig()
    config.CopyFrom(self._config)
    config.ClearField('topic_name')
    for topic_name in topic_names:
      config.topic_name.append(topic_name)
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

  @dictionary_name.setter
  def dictionary_name(self, dictionary_name):
    self._dictionary_name = dictionary_name
    config = artm.messages_pb2.SmoothSparsePhiConfig()
    config.CopyFrom(self._config)
    config.dictionary_name = dictionary_name
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

#######################################################################################################################
class SmoothSparseThetaRegularizer(object):
  """ SmoothSparseThetaRegularizer is a regularizer in ArtmModel (public class).
  Args:
  - name --- the identificator of regularizer. Is string, default = None
  - tau --- the coefficient of regularization for this regularizer, double, default = 1.0
  - topic_names --- list of names of topics to regularize. Is list of strings, default = None
  - alpha_iter --- list of additional coefficients of regularization on each iteration over document. 
                   Should have length equal to model.inner_iterations_count. Is list of double, default = None
  """
  def __init__(self, name=None, tau=1.0, topic_names=None, alpha_iter=None):
    config = artm.messages_pb2.SmoothSparseThetaConfig()
    self._topic_names = []
    self._alpha_iter = []

    if name is None:
      name = "SmoothSparseThetaRegularizer:" + uuid.uuid1().urn
    if not topic_names is None:
      config.ClearField('topic_name')
      for topic_name in topic_names:
        config.topic_name.append(topic_name)
        self._topic_names.append(topic_name)
    if not alpha_iter is None:
      config.ClearField('alpha_iter')
      for alpha in alpha_iter:
        config.alpha_iter.append(alpha)
        self._alpha_iter.append(alpha)

    self._name = name
    self._tau = tau
    self._config = config
    self._type = artm.library.RegularizerConfig_Type_SmoothSparseTheta
    self._regularizer = None  # reserve place for regularizer
    self._model = None  # reserve place for model
    
  @property
  def name(self): return self._name
  @property
  def tau(self): return self._tau 
  @property
  def config(self): return self._config  
  @property
  def type(self): return self._type
  @property
  def topic_names(self): return self._topic_names
  @property
  def alpha_iter(self): return self._alpha_iter
  @property
  def model(self): return self._model
  @property
  def regularizer(self): return self._regularizer


  @model.setter
  def model(self, model): self._model = model
  @regularizer.setter
  def regularizer(self, regularizer): self._regularizer = regularizer

  @tau.setter
  def tau(self, tau):
    self._tau = tau
    config = artm.messages_pb2.ModelConfig()
    config.CopyFrom(self._model.config())
    for i in range(len(config.regularizer_settings)):
      if config.regularizer_settings[i].name == self._name:
        config.regularizer_settings[i].tau = tau
        break
    self._model.Reconfigure(config)

  @topic_names.setter
  def topic_names(self, topic_names):
    self._topic_names = topic_names
    config = artm.messages_pb2.SmoothSparseThetaConfig()
    config.CopyFrom(self._config)
    config.ClearField('topic_name')
    for topic_name in topic_names:
      config.topic_name.append(topic_name)
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

  @alpha_iter.setter
  def alpha_iter(self, alpha_iter):
    self._alpha_iter = alpha_iter
    config = artm.messages_pb2.SmoothSparseThetaConfig()
    config.CopyFrom(self._config)
    config.ClearField('alpha_iter')
    for alpha in alpha_iter:
      config.alpha_iter.append(alpha)
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

#######################################################################################################################
class DecorrelatorPhiRegularizer(object):
  """ DecorrelatorPhiRegularizer is a regularizer in ArtmModel (public class).
  Args:
  - name --- the identificator of regularizer. Is string, default = None
  - tau --- the coefficient of regularization for this regularizer, double, default = 1.0
  - class_ids --- list of class_ids to regularize. Is list of strings, default = None
  - topic_names --- list of names of topics to regularize. Is list of strings, default = None
  """
  def __init__(self, name=None, tau=1.0, class_ids=None, topic_names=None):
    config = artm.messages_pb2.DecorrelatorPhiConfig()
    self._class_ids = []
    self._topic_names = []

    if name is None:
      name = "DecorrelatorPhiRegularizer:" + uuid.uuid1().urn
    if not class_ids is None:
      config.ClearField('class_id')
      for class_id in class_ids:
        config.class_id.append(class_id)
        self._class_ids.append(class_id)
    if not topic_names is None:
      config.ClearField('topic_name')
      for topic_name in topic_names:
        config.topic_name.append(topic_name)
        self._topic_names.append(topic_name)

    self._name = name
    self._tau = tau
    self._config = config
    self._type = artm.library.RegularizerConfig_Type_DecorrelatorPhi
    self._regularizer = None  # reserve place for regularizer
    self._model = None  # reserve place for model
    
  @property
  def name(self): return self._name
  @property
  def tau(self): return self._tau 
  @property
  def config(self): return self._config  
  @property
  def type(self): return self._type
  @property
  def class_ids(self): return self._class_ids
  @property
  def topic_names(self): return self._topic_names
  @property
  def model(self): return self._model
  @property
  def regularizer(self): return self._regularizer


  @model.setter
  def model(self, model): self._model = model
  @regularizer.setter
  def regularizer(self, regularizer): self._regularizer = regularizer

  @tau.setter
  def tau(self, tau):
    self._tau = tau
    config = artm.messages_pb2.ModelConfig()
    config.CopyFrom(self._model.config())
    for i in range(len(config.regularizer_settings)):
      if config.regularizer_settings[i].name == self._name:
        config.regularizer_settings[i].tau = tau
        break
    self._model.Reconfigure(config)

  @class_ids.setter
  def class_ids(self, class_ids):
    self._class_ids = class_ids
    config = artm.messages_pb2.DecorrelatorPhiConfig()
    config.CopyFrom(self._config)
    config.ClearField('class_id')
    for class_id in class_ids:
      config.class_id.append(class_id)
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

  @topic_names.setter
  def topic_names(self, topic_names):
    self._topic_names = topic_names
    config = artm.messages_pb2.DecorrelatorPhiConfig()
    config.CopyFrom(self._config)
    config.ClearField('topic_name')
    for topic_name in topic_names:
      config.topic_name.append(topic_name)
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

#######################################################################################################################
class LableRegularizationPhiRegularizer(object):
  """ LableRegularizationPhiRegularizer is a regularizer in ArtmModel (public class).
  Args:
  - name --- the identificator of regularizer. Is string, default = None
  - tau --- the coefficient of regularization for this regularizer, double, default = 1.0
  - class_ids --- list of class_ids to regularize. Is list of strings, default = None
  - topic_names --- list of names of topics to regularize. Is list of strings, default = None
  - dictionary --- BigARTM collection dictionary. Is string, default = None
  """
  def __init__(self, name=None, tau=1.0, class_ids=None, topic_names=None, dictionary_name=None):
    config = artm.messages_pb2.LableRegularizationPhiConfig()
    self._class_ids = []
    self._topic_names = []
    self._dictionary_name = ''

    if name is None:
      name = "LableRegularizationPhiRegularizer:" + uuid.uuid1().urn
    if not class_ids is None:
      config.ClearField('class_id')
      for class_id in class_ids:
        config.class_id.append(class_id)
        self._class_ids.append(class_id)
    if not topic_names is None:
      config.ClearField('topic_name')
      for topic_name in topic_names:
        config.topic_name.append(topic_name)
        self._topic_names.append(topic_name)
    if not dictionary_name is None:
      config.dictionary_name = dictionary_name
      self._dictionary_name = dictionary_name

    self._name = name
    self._tau = tau
    self._config = config
    self._type = artm.library.RegularizerConfig_Type_LableRegularizationPhi
    self._regularizer = None  # reserve place for regularizer
    self._model = None  # reserve place for model
    
  @property
  def name(self): return self._name
  @property
  def tau(self): return self._tau 
  @property
  def config(self): return self._config  
  @property
  def type(self): return self._type
  @property
  def class_ids(self): return self._class_ids
  @property
  def topic_names(self): return self._topic_names
  @property
  def dictionary_name(self): return self._dictionary_name
  @property
  def model(self): return self._model
  @property
  def regularizer(self): return self._regularizer


  @model.setter
  def model(self, model): self._model = model
  @regularizer.setter
  def regularizer(self, regularizer): self._regularizer = regularizer

  @tau.setter
  def tau(self, tau):
    self._tau = tau
    config = artm.messages_pb2.ModelConfig()
    config.CopyFrom(self._model.config())
    for i in range(len(config.regularizer_settings)):
      if config.regularizer_settings[i].name == self._name:
        config.regularizer_settings[i].tau = tau
        break
    self._model.Reconfigure(config)

  @class_ids.setter
  def class_ids(self, class_ids):
    self._class_ids = class_ids
    config = artm.messages_pb2.LableRegularizationPhiConfig()
    config.CopyFrom(self._config)
    config.ClearField('class_id')
    for class_id in class_ids:
      config.class_id.append(class_id)
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

  @topic_names.setter
  def topic_names(self, topic_names):
    self._topic_names = topic_names
    config = artm.messages_pb2.LableRegularizationPhiConfig()
    config.CopyFrom(self._config)
    config.ClearField('topic_name')
    for topic_name in topic_names:
      config.topic_name.append(topic_name)
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

  @dictionary_name.setter
  def dictionary_name(self, dictionary_name):
    self._dictionary_name = dictionary_name
    config = artm.messages_pb2.LableRegularizationPhiConfig()
    config.CopyFrom(self._config)
    config.dictionary_name = dictionary_name
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

#######################################################################################################################
class SpecifiedSparsePhiRegularizer(object):
  """ SpecifiedSparsePhiRegularizer is a regularizer in ArtmModel (public class).
  Args:
  - name --- the identificator of regularizer. Is string, default = None
  - tau --- the coefficient of regularization for this regularizer, double, default = 1.0
  - class_id --- class_id to regularize. Is string, default = None
  - topic_names --- list of names of topics to regularize. Is list of strings, default = None
  - max_elements_count --- number of elements to save in row/column. Is int, default = None
  - probability_threshold --- if m elements in row/column summarize into value >= probability_threshold, 
                              m < n => only these elements would be saved. Is double, in (0,1), default = None
  - sparse_by_columns --- find max elements in column or in row. Is bool, default = True
  """
  def __init__(self, name=None, tau=1.0, class_id=None, topic_names=None,
               max_elements_count=None, probability_threshold=None, sparse_by_columns=True):
    config = artm.messages_pb2.SpecifiedSparsePhiConfig()
    self._class_id = '@default_class'
    self._topic_names = []
    self._max_elements_count = 20
    self._probability_threshold = 0.99
    self._sparse_by_columns = True

    if name is None:
      name = "SpecifiedSparsePhiRegularizer:" + uuid.uuid1().urn
    if not class_id is None:
      config.class_id = class_id
      self._class_id = class_id
    if not topic_names is None:
      config.ClearField('topic_name')
      for topic_name in topic_names:
        config.topic_name.append(topic_name)
        self._topic_names.append(topic_name)
    if not max_elements_count is None:
      config.max_elements_count = max_elements_count
      self._max_elements_count = max_elements_count
    if not probability_threshold is None:
      config.probability_threshold = probability_threshold
      self._probability_threshold = probability_threshold
    if not sparse_by_columns is None:
      if sparse_by_columns == True:
        config.mode = artm.library.SpecifiedSparsePhiConfig_Mode_SparseTopics
        self._sparse_by_columns = True
      else:
        config.mode = artm.library.SpecifiedSparsePhiConfig_Mode_SparseTokens
        self._sparse_by_columns = False

    self._name = name
    self._tau = tau
    self._config = config
    self._type = artm.library.RegularizerConfig_Type_SpecifiedSparsePhi
    self._regularizer = None  # reserve place for regularizer
    self._model = None  # reserve place for model
    
  @property
  def name(self): return self._name
  @property
  def tau(self): return self._tau 
  @property
  def config(self): return self._config  
  @property
  def type(self): return self._type
  @property
  def class_id(self): return self._class_id
  @property
  def topic_names(self): return self._topic_names
  @property
  def max_elements_count(self): return self._max_elements_count
  @property
  def probability_threshold(self): return self._probability_threshold
  @property
  def sparse_by_columns(self): return self._sparse_by_columns
  @property
  def model(self): return self._model
  @property
  def regularizer(self): return self._regularizer


  @model.setter
  def model(self, model): self._model = model
  @regularizer.setter
  def regularizer(self, regularizer): self._regularizer = regularizer

  @tau.setter
  def tau(self, tau):
    self._tau = tau
    config = artm.messages_pb2.ModelConfig()
    config.CopyFrom(self._model.config())
    for i in range(len(config.regularizer_settings)):
      if config.regularizer_settings[i].name == self._name:
        config.regularizer_settings[i].tau = tau
        break
    self._model.Reconfigure(config)

  @class_id.setter
  def class_id(self, class_id):
    self._class_id = class_id
    config = artm.messages_pb2.SpecifiedSparsePhiConfig()
    config.CopyFrom(self._config)
    config.class_id = class_id
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

  @topic_names.setter
  def topic_names(self, topic_names):
    self._topic_names = topic_names
    config = artm.messages_pb2.SpecifiedSparsePhiConfig()
    config.CopyFrom(self._config)
    config.ClearField('topic_name')
    for topic_name in topic_names:
      config.topic_name.append(topic_name)
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

  @max_elements_count.setter
  def max_elements_count(self, max_elements_count):
    self._max_elements_count = max_elements_count
    config = artm.messages_pb2.SpecifiedSparseRegularizationPhiConfig()
    config.CopyFrom(self._config)
    config.max_elements_count = max_elements_count
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

  @probability_threshold.setter
  def probability_threshold(self, probability_threshold):
    self._probability_threshold = probability_threshold
    config = artm.messages_pb2.SpecifiedSparseRegularizationPhiConfig()
    config.CopyFrom(self._config)
    config.probability_threshold = probability_threshold
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

  @sparse_by_columns.setter
  def sparse_by_columns(self, sparse_by_columns):
    self._sparse_by_columns = sparse_by_columns
    config = artm.messages_pb2.SpecifiedSparseRegularizationPhiConfig()
    config.CopyFrom(self._config)
    if sparse_by_columns == True:
      config.mode = artm.library.SpecifiedSparsePhiConfig_Mode_SparseTopics
    else:
      config.mode = artm.library.SpecifiedSparsePhiConfig_Mode_SparseTokens
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

#######################################################################################################################
class ImproveCoherencePhiRegularizer(object):
  """ ImproveCoherencePhiRegularizer is a regularizer in ArtmModel (public class).
  Args:
  - name --- the identificator of regularizer. Is string, default = None
  - tau --- the coefficient of regularization for this regularizer, double, default = 1.0
  - class_ids --- list of class_ids to regularize. Is list of strings, default = None
  - topic_names --- list of names of topics to regularize. Is list of strings, default = None
  - dictionary --- BigARTM collection dictionary. Is string, default = None
  """
  def __init__(self, name=None, tau=1.0, class_ids=None, topic_names=None, dictionary_name=None):
    config = artm.messages_pb2.ImproveCoherencePhiConfig()
    self._class_ids = []
    self._topic_names = []
    self._dictionary_name = ''

    if name is None:
      name = "ImproveCoherencePhiRegularizer:" + uuid.uuid1().urn
    if not class_ids is None:
      config.ClearField('class_id')
      for class_id in class_ids:
        config.class_id.append(class_id)
        self._class_ids.append(class_id)
    if not topic_names is None:
      config.ClearField('topic_name')
      for topic_name in topic_names:
        config.topic_name.append(topic_name)
        self._topic_names.append(topic_name)
    if not dictionary_name is None:
      config.dictionary_name = dictionary_name
      self._dictionary_name = dictionary_name

    self._name = name
    self._tau = tau
    self._config = config
    self._type = artm.library.RegularizerConfig_Type_ImproveCoherencePhi
    self._regularizer = None  # reserve place for regularizer
    self._model = None  # reserve place for model
    
  @property
  def name(self): return self._name
  @property
  def tau(self): return self._tau 
  @property
  def config(self): return self._config  
  @property
  def type(self): return self._type
  @property
  def class_ids(self): return self._class_ids
  @property
  def topic_names(self): return self._topic_names
  @property
  def dictionary_name(self): return self._dictionary_name
  @property
  def model(self): return self._model
  @property
  def regularizer(self): return self._regularizer


  @model.setter
  def model(self, model): self._model = model
  @regularizer.setter
  def regularizer(self, regularizer): self._regularizer = regularizer

  @tau.setter
  def tau(self, tau):
    self._tau = tau
    config = artm.messages_pb2.ModelConfig()
    config.CopyFrom(self._model.config())
    for i in range(len(config.regularizer_settings)):
      if config.regularizer_settings[i].name == self._name:
        config.regularizer_settings[i].tau = tau
        break
    self._model.Reconfigure(config)

  @class_ids.setter
  def class_ids(self, class_ids):
    self._class_ids = class_ids
    config = artm.messages_pb2.ImproveCoherencePhiConfig()
    config.CopyFrom(self._config)
    config.ClearField('class_id')
    for class_id in class_ids:
      config.class_id.append(class_id)
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

  @topic_names.setter
  def topic_names(self, topic_names):
    self._topic_names = topic_names
    config = artm.messages_pb2.ImproveCoherencePhiConfig()
    config.CopyFrom(self._config)
    config.ClearField('topic_name')
    for topic_name in topic_names:
      config.topic_name.append(topic_name)
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

  @dictionary_name.setter
  def dictionary_name(self, dictionary_name):
    self._dictionary_name = dictionary_name
    config = artm.messages_pb2.ImproveCoherencePhiConfig()
    config.CopyFrom(self._config)
    config.dictionary_name = dictionary_name
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

#######################################################################################################################
# SECTION OF SCORE CLASSES
#######################################################################################################################
class SparsityPhiScore(object):
  """ SparsityPhiScore is a score in ArtmModel (public class).
  Args:
  - name --- the identificator of score. Is string, default = None
  - class_id --- class_id to score. Is string, default = None
  - topic_names --- list of names of topics to score. Is list of strings, default = None
  - eps --- the tolerance const, everything < eps considered to be zero. Is double, default = None
  """
  def __init__(self, name=None, class_id=None, topic_names=None, eps=None):
    config = artm.messages_pb2.SparsityPhiScoreConfig()
    self._class_id = '@default_class'
    self._topic_names = []
    self._eps = 1e-37

    if name is None:
      name = "SparsityPhiScore:" + uuid.uuid1().urn
    if not class_id is None:
      config.class_id = class_id
      self._class_id = class_id
    if not topic_names is None:
      config.ClearField('topic_name')
      for topic_name in topic_names:
        config.topic_name.append(topic_name)
        self._topic_names.append(topic_name)
    if not eps is None:
      self._eps = eps

    self._name = name
    self._config = config
    self._type = artm.library.ScoreConfig_Type_SparsityPhi
    self._model = None  # reserve place for model
    self._master = None  # reserve place for master (to reconfigure Scores)
    self._score = None  # reserve place for score
    
  @property
  def name(self): return self._name
  @property
  def config(self): return self._config  
  @property
  def type(self): return self._type
  @property
  def class_id(self): return self._class_id
  @property
  def topic_names(self): return self._topic_names
  @property
  def eps(self): return self._eps
  @property
  def model(self): return self._model
  @property
  def score(self): return self._score
  @property
  def master(self): return self._master

  @model.setter
  def model(self, model): self._model = model
  @score.setter
  def score(self, score): self._score = score
  @master.setter
  def master(self, master): self._master = master

  @class_id.setter
  def class_id(self, class_id):
    self._class_id = class_id
    score_config = artm.messages_pb2.SparsityPhiScoreConfig()
    score_config.CopyFrom(self._config)
    score_config.class_id = class_id

    master_config = artm.messages_pb2.MasterComponentConfig()
    master_config.CopyFrom(self._master.config())
    for i in range(len(master_config.score_config)):
      if master_config.score_config[i].name == self._name:
        master_config.score_config[i].config = score_config.SerializeToString()
        break
    self._master.Reconfigure(master_config)

  @topic_names.setter
  def topic_names(self, topic_names):
    self._topic_names = topic_names
    score_config = artm.messages_pb2.SparsityPhiScoreConfig()
    score_config.CopyFrom(self._config)
    score_config.ClearField('topic_names')
    for topic_name in topic_names:
      score_config.topic_name.append(topic_name)

    master_config = artm.messages_pb2.MasterComponentConfig()
    master_config.CopyFrom(self._master.config())
    for i in range(len(master_config.score_config)):
      if master_config.score_config[i].name == self._name:
        master_config.score_config[i].config = score_config.SerializeToString()
        break
    self._master.Reconfigure(master_config)

  @eps.setter
  def eps(self, eps):
    self._eps = eps
    score_config = artm.messages_pb2.SparsityPhiScoreConfig()
    score_config.CopyFrom(self._config)
    score_config.eps = eps

    master_config = artm.messages_pb2.MasterComponentConfig()
    master_config.CopyFrom(self._master.config())
    for i in range(len(master_config.score_config)):
      if master_config.score_config[i].name == self._name:
        master_config.score_config[i].config = score_config.SerializeToString()
        break
    self._master.Reconfigure(master_config)

#######################################################################################################################
# SECTION OF SCORE INFO CLASSES
#######################################################################################################################
class SparsityPhiScoreInfo(object):
  """ SparsityPhiScoreInfo represents a result of counting SparsityPhiScore (private class).
  Args:
  - score --- reference to score object, no default
  """
  def __init__(self, score):   
    self._name = score.name
    self._value = []
    self._zero_tokens = []
    self._total_tokens = []

  """ SparsityPhiScoreInfo.add() --- add info about score after syncronization.
  Args:
  - score --- reference to score object, default = None (means "Add None values")
  """
  def add(self, score=None):
    if not score is None:
      _data = artm.messages_pb2.SparsityPhiScore()
      _data = score.score.GetValue(score._model)
    
      self._value.append(_data.value)
      self._zero_tokens.append(_data.zero_tokens)
      self._total_tokens.append(_data.total_tokens)
    else:
      self._value.append(None)
      self._zero_tokens.append(None)
      self._total_tokens.append(None)

  @property
  def name(self): return self._name
  @property
  def value(self): return self._value  
  @property
  def zero_tokens(self): return self._zero_tokens
  @property
  def total_tokens(self): return self._total_tokens
 
#######################################################################################################################
