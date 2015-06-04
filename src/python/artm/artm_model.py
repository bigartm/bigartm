import artm.messages_pb2 as messages_pb2
import artm.library as library
import collections
from collections import OrderedDict
from collections import namedtuple
import random
import uuid
import sys
import shutil
import os
from os import path
import glob
import csv

# forward declaration
class ArtmModel(object): pass
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
ThetaRegularizer_Type = 0
PhiRegularizer_Type = 1

def reconfigure_score_in_master(master, score_config, name):
  master_config = messages_pb2.MasterComponentConfig()
  master_config.CopyFrom(master.config())
  for i in range(len(master_config.score_config)):
    if master_config.score_config[i].name == name:
      master_config.score_config[i].config = score_config.SerializeToString()
      break
  master.Reconfigure(master_config)

def create_parser_config(data_path, collection_name, target_folder):
  collection_parser_config = messages_pb2.CollectionParserConfig()
  collection_parser_config.format = library.CollectionParserConfig_Format_BagOfWordsUci
  collection_parser_config.num_items_per_batch = batch_size
  collection_parser_config.docword_file_path = data_path + 'docword.' + collection_name + '.txt'
  collection_parser_config.vocab_file_path = data_path + 'vocab.' + collection_name + '.txt'
  collection_parser_config.target_folder = target_folder
  collection_parser_config.dictionary_file_name = 'dictionary'
  return collection_parser_config

#######################################################################################################################
class ArtmModel(object):
  """ ArtmModel represents a topic model (public class).
  Args:
  - num_processors --- how many threads will be used for model training. Is int, default = 1
  - topic_names --- names of topics in model. Is list of strings, default = []
  - topics_count --- number of topics in model (is used if topic_names == []). Is int, default = 10
  - class_ids --- list of class_ids and their weights to be used in model. Is dict, 
                  key --- class_id, value --- weight, default = {}
  - document_passes_count --- number of iterations over each document during processing/ Is int, default = 1
  - cache_theta --- save or not the Theta matrix in model. Necessary if ArtmModel.get_theta() usage expects. 
                    Is bool, default = True
  Important public fields:
  - regularizers --- contains dict of regularizers, included into model
  - scores --- contains dict of scores, included into model
  - scores_info --- contains dict of scoring results; key --- score name, value --- ScoreInfo object, which 
                    contains info about values of score on each synchronization in list
  NOTE:
  Here and anywhere in BigARTM empty topic_names or class_ids means that model (or regularizer, or score) should 
  use all topics or class_ids. If some fields of regularizers or scores are not defined by user --- internal library 
  defaults would be used.
  """

######### CONSTRUCTOR #########
  def __init__(self, num_processors=1, topic_names=[], topics_count=10,
               class_ids={}, document_passes_count=1, cache_theta=True):
    self._num_processors = 1
    self._topics_count = 10
    self._topic_names = []
    self._class_ids = {}
    self._document_passes_count = 1
    self._cache_theta = True

    if num_processors > 0:            self._num_processors        = num_processors
    if topics_count > 0:              self._topics_count          = topics_count
    if len(topic_names) > 0:          self._topic_names           = topic_names
    if len(class_ids) > 0:            self._class_ids             = class_ids
    if document_passes_count > 0:     self._document_passes_count = document_passes_count
    if isinstance(cache_theta, bool): self._cache_theta           = cache_theta

    self._master = library.MasterComponent()
    self._master.config().processors_count = self._num_processors
    self._master.config().cache_theta = cache_theta
    self._master.Reconfigure()     

    self._model = 'pwt'
    self._regularizers = Regularizers(self._master)
    self._scores = Scores(self._master, self._model)

    self._scores_info = {}
    self._synchronizations_processed = -1 
    self._was_initialized = False

######### PROPERTIES #########
  @property
  def num_processors(self): return self._num_processors
  @property
  def document_passes_count(self): return self._document_passes_count
  @property
  def cache_theta(self): return self._cache_theta
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
  @property
  def master(self): return self._master

######### SETTERS #########
  @num_processors.setter
  def num_processors(self, num_processors):
    if num_processors <= 0 or not isinstance(num_processors, int):
      print 'Number of processors should be a positive integer, skip update'
    else:
      self._num_processors = num_processors
      self._master.config().processors_count = num_processors
      self._master.Reconfigure()
      
  @document_passes_count.setter
  def document_passes_count(self, document_passes_count):
    if document_passes_count <= 0 or not isinstance(document_passes_count, int):
      print 'Number of passes throug documents should be a positive integer, skip update'
    else:
      self._document_passes_count = document_passes_count

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

  @class_ids.setter
  def class_ids(self, class_ids):
    if len(class_ids) < 0:
      print 'Number of (class_id, class_weight) pairs shoul be non-negative, skip update'
    else:
      self._class_ids = class_ids

######### METHODS #########
  def parse(self, collection_name=None, data_path='', data_format='batches',
            batch_size=1000, gather_cooc=False, cooc_tokens=[]):
    """ ArtmModel.fit() --- proceed the learning of topic model
    Args:
    - collection_name --- the name of text collection (required if data_format == 'bow_uci'). 
                          Is string, default = None
    - data_path --- 1) if data_format == 'bow_uci' => folder containing docword.collection_name.txt 
                                                      and vocab.collection_name.txt files
                    2) if data_format == 'bow_vw' => file in Vowpal Wabbit format
                    3) if data_format == 'plain_text' => file with text
                    Is string, default = ''
    - data_format --- the type of input data: 1) 'bow_uci' --- Bag-Of-Words in UCI format
                                              2) 'bow_vw' --- Bag-Of-Words in Vowpal Wabbit format
                                              3) 'plain_text' --- source text
                                              Is string, default = 'bow_uci'
    - batch_size --- number of documents to be stored in each batch. Is int, default = 1000
    - gather_cooc --- find or not the info about the token pairwise co-occuracies. Is bool, default=False
    - cooc_tokens --- tokens to collect cooc info (has sense if gather_cooc is True). Is list of lists, each 
                      internal list represents token and contain two strings --- token and its class_id, default = []
    """
    if collection_name is None and data_format == 'bow_uci':
        print 'No collection name was given, skip model.fit_offline()'

    if data_format == 'bow_uci':
      collection_parser_config = create_parser_config(data_path, collection_name, collection_name)
      collection_parser_config.gather_cooc = gather_cooc
      for token in cooc_tokens:
        collection_parser_config.cooccurrence_token.append(token)
      unique_tokens = library.Library().ParseCollection(collection_parser_config)

    elif data_format == 'bow_vw':
      raise NotImplementedError()
    elif data_format == 'plain_text':
      raise NotImplementedError()
    else:
      print 'Unknown data format, skip model.parse()'


  def load_dictionary(self, dictionary_path=None):
    """ ArtmModel.load_dictionary() --- load and return the BigARTM dictionary of the collection
    Args:
    - dictionary_path --- full file name of the dictionary. Is string, default = None
    """
    if not dictionary_path is None:
      unique_tokens = library.Library().LoadDictionary(dictionary_path)
      return self._master.CreateDictionary(unique_tokens)
    else:
      print 'dictionary path is None, skip loading dictionary.'


  def fit_offline(self, collection_name=None, batches=None, data_path='', collection_passes_count=1, decay_weight=0.9,
                  apply_weight=0.1, reset_theta_scores=False, data_format='batches', batch_size=1000):
    """ ArtmModel.fit_offline() --- proceed the learning of topic model in off-line mode
    Args:
    - collection_name --- the name of text collection (required if data_format == 'bow_uci'). 
                          Is string, default = None
    - batches --- list of file names of batches to be processed. If not None, than data_format should be 'batches'.
                  Is list of strings in format '*.batch', default = None
    - data_path --- 1) if data_format == 'batches' => folder containing batches and dictionary
                    2) if data_format == 'bow_uci' => folder containing docword.collection_name.txt 
                                                      and vocab.collection_name.txt files
                    3) if data_format == 'bow_vw' => file in Vowpal Wabbit format
                    4) if data_format == 'plain_text' => file with text
                    Is string, default = ''
    - collection_passes_count --- number of iterations over whole given collection. Is int, default = 1
    - decay_weight --- coefficient for applying old n_wt counters. Is int, default = 0.9
    - apply_weight --- coefficient for applying new n_wt counters. Is int, default = 0.1
    - reset_theta_scores --- reset accumulated Theta scores before learning. Is bool, default = False
    - data_format --- the type of input data: 1) 'batches' --- the data in format of BigARTM
                                              2) 'bow_uci' --- Bag-Of-Words in UCI format
                                              3) 'bow_vw' --- Bag-Of-Words in Vowpal Wabbit format
                                              4) 'plain_text' --- source text
                                              Is string, default = 'batches'
    Next argument has sense only if data_format is not 'batches' (e.g. parsing is necessary).
    - batch_size --- number of documents to be stored in each batch. Is int, default = 1000
    Note: ArtmModel.init() should be proceed before first call ArtmModel.fit_offline(),
          or it will be initialized by dictionary during first call.
    """
    if collection_name is None and data_format == 'bow_uci':
      print 'No collection name was given, skip model.fit_offline()'

    if not data_format == 'batches' and not batches is None:
      print "batches != None require data_format == 'batches'" 

    unique_tokens = messages_pb2.DictionaryConfig()
    target_folder = data_path + '/batches_temp_' + str(random.uniform(0,1))
    batches_list = []
    if data_format == 'batches':
      if batches is None:
        batches_list = glob.glob(data_path + "/*.batch")
        if len(batches_list) < 1:
          print 'No batches were found, skip model.fit()'
          return
        print 'Found ' + str(len(batches_list)) + ' batches, using them.'
        unique_tokens = library.Library().LoadDictionary(data_path + '/dictionary')
      else:
        batches_list = [data_path + '/' + batch for batch in batches]

    elif data_format == 'bow_uci':
      collection_parser_config = create_parser_config(data_path, collection_name, target_folder)
      unique_tokens = library.Library().ParseCollection(collection_parser_config)
      batches_list = glob.glob(target_folder + "/*.batch")

    elif data_format == 'bow_vw':
      raise NotImplementedError()
    elif data_format == 'plain_text':
      raise NotImplementedError()
    else:
      print 'Unknown data format, skip model.fit_offline()'

    if not self._was_initialized:
      self.init(dictionary=self._master.CreateDictionary(unique_tokens))
      self._was_initialized = True
      
    theta_regularizers, phi_regularizers = {}, {}
    for name, config in self._regularizers.data.iteritems():
      if config.type == ThetaRegularizer_Type: theta_regularizers[name] = config.tau
      else:                                    phi_regularizers[name] = config.tau

    for iter in range(collection_passes_count):
      self._master.ProcessBatches(pwt=self._model,
                                  batches=batches_list,
                                  target_nwt='nwt_hat',
                                  regularizers=theta_regularizers,
                                  inner_iterations_count=self._document_passes_count,
                                  class_ids=self._class_ids,
                                  reset_scores=reset_theta_scores)
      self._synchronizations_processed +=1
      if self._synchronizations_processed == 0:
        self._master.MergeModel({self._model : decay_weight, 'nwt_hat' : apply_weight},
                                target_nwt='nwt', topic_names=self._topic_names)
      else:
        self._master.MergeModel({'nwt' : decay_weight, 'nwt_hat' : apply_weight},
                                target_nwt='nwt', topic_names=self._topic_names)

      self._master.RegularizeModel(self._model, 'nwt', 'rwt', phi_regularizers)
      self._master.NormalizeModel('nwt', self._model, 'rwt')

      for name in self.scores.data.keys():
        if not name in self.scores_info:
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

          for i in range(self._synchronizations_processed):
            self._scores_info[name].add()

        self._scores_info[name].add(self.scores[name])
              
    # remove temp batches folder if necessary
    if not data_format == 'batches':
      shutil.rmtree(target_folder)


  def fit_online(self, collection_name=None, batches=None, data_path='', tau0=1024.0, kappa=0.7, update_every=1,
                 reset_theta_scores=False, data_format='batches', batch_size=1000):
    """ ArtmModel.fit_online() --- proceed the learning of topic model in on-line mode
    Args:
    - collection_name --- the name of text collection (required if data_format == 'bow_uci'). 
                          Is string, default = None
    - batches --- list of file names of batches to be processed. If not None, than data_format should be 'batches'.
                  Is list of strings in format '*.batch', default = None
    - data_path --- 1) if data_format == 'batches' => folder containing batches and dictionary
                    2) if data_format == 'bow_uci' => folder containing docword.collection_name.txt 
                                                      and vocab.collection_name.txt files
                    3) if data_format == 'bow_vw' => file in Vowpal Wabbit format
                    4) if data_format == 'plain_text' => file with text
                    Is string, default = ''
    - update_every --- the number of batches; model will be updated once per it. Is int, default = 1
    - tau0 --- coefficient (see kappa). Is float, default = 1024.0
    - kappa --- power for tau0. Is float, default = 0.7
                The formulas for decay_weight and apply_weight:
            update_count = current_processed_documents / (batch_size * update_every)
            rho = pow(tau0 + update_count, -kappa)
            decay_weight = 1-rho
            apply_weight = rho
    - reset_theta_scores --- reset accumulated Theta scores before learning. Is bool, default = False
    - data_format --- the type of input data: 1) 'batches' --- the data in format of BigARTM
                                              2) 'bow_uci' --- Bag-Of-Words in UCI format
                                              3) 'bow_vw' --- Bag-Of-Words in Vowpal Wabbit format
                                              4) 'plain_text' --- source text
                                              Is string, default = 'batches'
    Next argument has sense only if data_format is not 'batches' (e.g. parsing is necessary).
    - batch_size --- number of documents to be stored in each batch. Is int, default = 1000
    Note: ArtmModel.init() should be proceed before first call ArtmModel.fit_online(),
          or it will be initialized by dictionary during first call.
    """
    if collection_name is None and data_format == 'bow_uci':
        print 'No collection name was given, skip model.fit_online()'

    if not data_format == 'batches' and not batches is None:
      print "batches != None require data_format == 'batches'"

    unique_tokens = messages_pb2.DictionaryConfig()
    target_folder = data_path + '/batches_temp_' + str(random.uniform(0,1))
    batches_list = []
    if data_format == 'batches':
      if batches is None:
        batches_list = glob.glob(data_path + "/*.batch")
        if len(batches_list) < 1:
          print 'No batches were found, skip model.fit()'
          return
        print 'Found ' + str(len(batches_list)) + ' batches, using them.'
        unique_tokens = library.Library().LoadDictionary(data_path + '/dictionary')
      else:
        batches_list = [data_path + '/' + batch for batch in batches]

    elif data_format == 'bow_uci':
      collection_parser_config = create_parser_config(data_path, collection_name, target_folder)
      unique_tokens = library.Library().ParseCollection(collection_parser_config)
      batches = glob.glob(target_folder + "/*.batch")

    elif data_format == 'bow_vw':
      raise NotImplementedError()
    elif data_format == 'plain_text':
      raise NotImplementedError()
    else:
      print 'Unknown data format, skip model.fit_online()'

    if not self._was_initialized:
      self.init(dictionary=self._master.CreateDictionary(unique_tokens))
      self._was_initialized = True

    theta_regularizers, phi_regularizers = {}, {}
    for name, config in self._regularizers.data.iteritems():
      if config.type == ThetaRegularizer_Type: theta_regularizers[name] = config.tau
      else:                                    phi_regularizers[name] = config.tau

    batches_to_process = []
    current_processed_documents = 0
    for batch_index, batch_filename in enumerate(batches_list):
      batches_to_process.append(batch_filename)
      if ((batch_index + 1) % update_every == 0) or ((batch_index + 1) == len(batches_list)):
        self._master.ProcessBatches(pwt=self._model,
                                    batches=batches_to_process,
                                    target_nwt='nwt_hat',
                                    regularizers=theta_regularizers,
                                    inner_iterations_count=self._document_passes_count,
                                    class_ids=self._class_ids,
                                    reset_scores=reset_theta_scores)

        current_processed_documents += batch_size * update_every
        update_count = current_processed_documents / (batch_size * update_every)
        rho = pow(tau0 + update_count, -kappa)
        decay_weight, apply_weight = 1-rho, rho
        
        self._synchronizations_processed +=1
        if self._synchronizations_processed == 0:
          self._master.MergeModel({self._model : decay_weight, 'nwt_hat' : apply_weight},
                                  target_nwt='nwt', topic_names=self._topic_names)
        else:
          self._master.MergeModel({'nwt' : decay_weight, 'nwt_hat' : apply_weight},
                                  target_nwt='nwt', topic_names=self._topic_names)            

        self._master.RegularizeModel(self._model, 'nwt', 'rwt', phi_regularizers)
        self._master.NormalizeModel('nwt', self._model, 'rwt')
        batches_to_process = []

        for name in self.scores.data.keys():
          if not name in self.scores_info:
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

            for i in range(self._synchronizations_processed):
              self._scores_info[name].add()

          self._scores_info[name].add(self.scores[name])
         
    # remove temp batches folder if necessary
    if not data_format == 'batches':
      shutil.rmtree(target_folder)


  def save(self, file_name='artm_model'):
    """ ArtmModel.save() --- save the topic model to disk.
    Args:
    - file_name --- the name of file to store model. Is string, default = 'artm_model'
    """
    if os.path.isfile(file_name): os.remove(file_name)
    self._master.ExportModel(self._model, file_name)


  def load(self, file_name):
    """ ArtmModel.load() --- load the topic model, saved by ArtmModel.save(), from disk.
    Args:
    - file_name --- the name of file containing model. Is string, no default
    Note: Loaded model will overwrite ArtmModel.topic_names and ArtmModel.topics_count fields.
          Also it will empty ArtmModel.scores_info.
    """
    self._master.ImportModel(self._model, file_name)
    self._was_initialized = True
    args = messages_pb2.GetTopicModelArgs()
    args.request_type = library.GetTopicModelArgs_RequestType_TopicNames
    topic_model = self._master.GetTopicModel(model=self._model, args=args)
    self._topic_names = [topic_name for topic_name in topic_model.topic_name]
    self._topics_count = topic_model.topics_count
    
    # remove all info about previous iterations
    self._scores_info = {}
    self._synchronizations_processed = -1


  def to_csv(self, file_name='artm_model.csv'):
    """ ArtmModel.to_csv() --- save the topic model to disk in .csv format (can't be loaded back).
    Args:
    - file_name --- the name of file to store model. Is string, default = 'artm_model.csv'
    """
    if os.path.isfile(file_name): os.remove(file_name)
    with open(file_name, 'wb') as csvfile:
      writer = csv.writer(csvfile, delimiter=';', quotechar=';', quoting=csv.QUOTE_MINIMAL)
      if len(self._topic_names) > 0:
        writer.writerow(['Token'] + ['Class ID'] + ['TOPIC: ' + topic_name for topic_name in self._topic_names])
      else:
        writer.writerow(['Token'] + ['Class ID'] + ['TOPIC: ' + index for index in range(self._topics_count)])
      topic_model = self._master.GetTopicModel(self._model)
      for index in range(len(topic_model.token_weights)):
         writer.writerow([topic_model.token[index]] + [topic_model.class_id[index]] +
                         [token_weight for token_weight in topic_model.token_weights[index].value])

  def get_theta(self, remove_theta=False):
    """ ArtmModel.get_theta() --- get Theta matrix for training set of documents.
    Args:
    - remove_theta --- flag indicates save or remove Theta from model after extraction. Is bool, default = False
    Output: triple (document_ids, topic_names, values), where:
            1) document_ids --- the ids of documents, for which the Theta matrix was requested. Is list of strings
            2) topic_names --- the names of topics in topic model, that was used to create Theta. Is list of strings
            3) values --- content of Theta matrix. Is list of lists of floats, each internal list represents one
               document and has length equal to len(topic_names).
    """
    if self.cache_theta == False:
      print 'ArtmModel.cache_theta == False, skip get_theta(). Set ArtmModel.cache_theta = True'
    else:
      theta_matrix = self._master.GetThetaMatrix(self._model, clean_cache=remove_theta)
      document_ids = [item_id for item_id in theta_matrix.item_id]
      topic_names = [topic_name for topic_name in theta_matrix.topic_name]
      values = [[item_weight for item_weight in item_weights.value] for item_weights in theta_matrix.item_weights]

      return (document_ids, topic_names, values)


  def find_theta(self, batches=None, collection_name=None, data_path='', data_format='batches'):
    """ ArtmModel.find_theta() --- find Theta matrix for new documents.
    Args:
    - collection_name --- the name of text collection (required if data_format == 'bow_uci'). 
                          Is string, default = None
    - batches --- list of file names of batches to be processed. If not None, than data_format should be 'batches'.
                  Is list of strings in format '*.batch', default = None
    - data_path --- 1) if data_format == 'batches' => folder containing batches and dictionary
                    2) if data_format == 'bow_uci' => folder containing docword.txt and vocab.txt files
                    3) if data_format == 'bow_vw' => file in Vowpal Wabbit format
                    4) if data_format == 'plain_text' => file with text
                    Is string, default = ''
    - data_format --- the type of input data: 1) 'batches' --- the data in format of BigARTM
                                              2) 'bow_uci' --- Bag-Of-Words in UCI format
                                              3) 'bow_vw' --- Bag-Of-Words in Vowpal Wabbit format
                                              4) 'plain_text' --- source text
                                              Is string, default = 'batches'
    Output: triple (document_ids, topic_names, values), where:
            1) document_ids --- the ids of documents, for which the Theta matrix was requested. Is list of strings
            2) topic_names --- the names of topics in topic model, that was used to create Theta. Is list of strings
            3) values --- content of Theta matrix. Is list of lists of floats, each internal list represents one
               document and has length equal to len(topic_names).
    """
    if collection_name is None and data_format == 'bow_uci':
      print 'No collection name was given, skip model.fit_offline()'

    if not data_format == 'batches' and not batches is None:
      print "batches != None require data_format == 'batches'" 

    target_folder = data_path + '/batches_temp_' + str(random.uniform(0,1))
    batches_list = []
    if data_format == 'batches':
      if batches is None:
        batches_list = glob.glob(data_path + "/*.batch")
        if len(batches_list) < 1:
          print 'No batches were found, skip model.fit()'
          return
        print 'Found ' + str(len(batches_list)) + ' batches, using them.'
      else:
        batches_list = [data_path + '/' + batch for batch in batches]

    elif data_format == 'bow_uci':
      collection_parser_config = create_parser_config(data_path, collection_name, target_folder)
      unique_tokens = library.Library().ParseCollection(collection_parser_config)
      batches_list = glob.glob(target_folder + "/*.batch")

    elif data_format == 'bow_vw':
      raise NotImplementedError()
    elif data_format == 'plain_text':
      raise NotImplementedError()
    else:
      print 'Unknown data format, skip model.find_theta()'
 
    results = self._master.ProcessBatches(pwt=self._model,
                                          batches=batches_list,
                                          target_nwt='nwt_hat',
                                          inner_iterations_count=self._document_passes_count,
                                          class_ids=self._class_ids,
                                          theta_matrix_type=library.ProcessBatchesArgs_ThetaMatrixType_Dense)
    theta_matrix = results.theta_matrix
    document_ids = [item_id for item_id in theta_matrix.item_id]
    topic_names = [topic_name for topic_name in theta_matrix.topic_name]
    values = [[item_weight for item_weight in item_weights.value] for item_weights in theta_matrix.item_weights]

    # remove temp batches folder if necessary
    if not data_format == 'batches':
      shutil.rmtree(target_folder)

    return (document_ids, topic_names, values)


  def init(self, data_path=None, dictionary=None):
    """ ArtmModel.init() --- initialize topic model before learning.
    Args:
    - data_path --- name of directory containing BigARTM batches. Is string, default = None
    - dictionary --- BigARTM collection dictionary. Is string, default = None
    Priority of initialization:
    1) batches in 'data_path'
    2) dictionary
    """
    if not data_path is None:
      self._master.InitializeModel(model_name=self._model, batch_folder=data_path,
                                   topics_count=self._topics_count, topic_names=self._topic_names)
    else:
      self._master.InitializeModel(model_name=self._model, dictionary=dictionary,
                                   topics_count=self._topics_count, topic_names=self._topic_names)   
    self._was_initialized = True                                   

#######################################################################################################################  
class Regularizers(object):
  """ Regularizers represents a storage of regularizers in ArtmModel (private class).
  Args:
  - master --- reference to master component object, no default
  """
  def __init__(self, master):
    self._data = {}
    self._master = master

  def add(self, config):
    """ Regularizers.add() --- add regularizer into ArtmModel.
    Args:
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
    Args:
    - name --- name of the regularizer. Is string, no default
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
  - name --- the identifier of regularizer. Is string, default = None
  - tau --- the coefficient of regularization for this regularizer, double, default = 1.0
  - class_ids --- list of class_ids to regularize. Is list of strings, default = None
  - topic_names --- list of names of topics to regularize. Is list of strings, default = None
  - dictionary_name --- BigARTM collection dictionary. Is string, default = None
  """
  def __init__(self, name=None, tau=1.0, class_ids=None, topic_names=None, dictionary_name=None):
    config = messages_pb2.SmoothSparsePhiConfig()
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
    self._type = PhiRegularizer_Type
    self._config = config
    self._type = library.RegularizerConfig_Type_SmoothSparsePhi
    self._regularizer = None  # reserve place for regularizer
    
  @property
  def name(self): return self._name
  @property
  def tau(self): return self._tau 
  @property
  def type(self): return self._type   
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
  def regularizer(self): return self._regularizer

  @regularizer.setter
  def regularizer(self, regularizer): self._regularizer = regularizer

  @tau.setter
  def tau(self, tau): self._tau = tau

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

#######################################################################################################################
class SmoothSparseThetaRegularizer(object):
  """ SmoothSparseThetaRegularizer is a regularizer in ArtmModel (public class).
  Args:
  - name --- the identifier of regularizer. Is string, default = None
  - tau --- the coefficient of regularization for this regularizer, double, default = 1.0
  - topic_names --- list of names of topics to regularize. Is list of strings, default = None
  - alpha_iter --- list of additional coefficients of regularization on each iteration over document. 
                   Should have length equal to model.document_passes_count. Is list of double, default = None
  """
  def __init__(self, name=None, tau=1.0, topic_names=None, alpha_iter=None):
    config = messages_pb2.SmoothSparseThetaConfig()
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
    self._type = ThetaRegularizer_Type
    self._config = config
    self._type = library.RegularizerConfig_Type_SmoothSparseTheta
    self._regularizer = None  # reserve place for regularizer
    
  @property
  def name(self): return self._name
  @property
  def tau(self): return self._tau 
  @property
  def type(self): return self._type 
  @property
  def config(self): return self._config  
  @property
  def type(self): return self._type
  @property
  def topic_names(self): return self._topic_names
  @property
  def alpha_iter(self): return self._alpha_iter
  @property
  def regularizer(self): return self._regularizer

  @regularizer.setter
  def regularizer(self, regularizer): self._regularizer = regularizer

  @tau.setter
  def tau(self, tau): self._tau = tau

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

#######################################################################################################################
class DecorrelatorPhiRegularizer(object):
  """ DecorrelatorPhiRegularizer is a regularizer in ArtmModel (public class).
  Args:
  - name --- the identifier of regularizer. Is string, default = None
  - tau --- the coefficient of regularization for this regularizer, double, default = 1.0
  - class_ids --- list of class_ids to regularize. Is list of strings, default = None
  - topic_names --- list of names of topics to regularize. Is list of strings, default = None
  """
  def __init__(self, name=None, tau=1.0, class_ids=None, topic_names=None):
    config = messages_pb2.DecorrelatorPhiConfig()
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
    self._type = PhiRegularizer_Type
    self._config = config
    self._type = library.RegularizerConfig_Type_DecorrelatorPhi
    self._regularizer = None  # reserve place for regularizer
    
  @property
  def name(self): return self._name
  @property
  def tau(self): return self._tau 
  @property
  def type(self): return self._type   
  @property
  def config(self): return self._config  
  @property
  def type(self): return self._type
  @property
  def class_ids(self): return self._class_ids
  @property
  def topic_names(self): return self._topic_names
  @property
  def regularizer(self): return self._regularizer

  @regularizer.setter
  def regularizer(self, regularizer): self._regularizer = regularizer

  @tau.setter
  def tau(self, tau): self._tau = tau

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

#######################################################################################################################
class LableRegularizationPhiRegularizer(object):
  """ LableRegularizationPhiRegularizer is a regularizer in ArtmModel (public class).
  Args:
  - name --- the identifier of regularizer. Is string, default = None
  - tau --- the coefficient of regularization for this regularizer, double, default = 1.0
  - class_ids --- list of class_ids to regularize. Is list of strings, default = None
  - topic_names --- list of names of topics to regularize. Is list of strings, default = None
  - dictionary_name --- BigARTM collection dictionary. Is string, default = None
  """
  def __init__(self, name=None, tau=1.0, class_ids=None, topic_names=None, dictionary_name=None):
    config = messages_pb2.LableRegularizationPhiConfig()
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
    self._type = PhiRegularizer_Type
    self._config = config
    self._type = library.RegularizerConfig_Type_LableRegularizationPhi
    self._regularizer = None  # reserve place for regularizer
    
  @property
  def name(self): return self._name
  @property
  def tau(self): return self._tau 
  @property
  def type(self): return self._type   
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
  def regularizer(self): return self._regularizer

  @regularizer.setter
  def regularizer(self, regularizer): self._regularizer = regularizer

  @tau.setter
  def tau(self, tau): self._tau = tau

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

#######################################################################################################################
class SpecifiedSparsePhiRegularizer(object):
  """ SpecifiedSparsePhiRegularizer is a regularizer in ArtmModel (public class).
  Args:
  - name --- the identifier of regularizer. Is string, default = None
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
    config = messages_pb2.SpecifiedSparsePhiConfig()
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
        config.mode = library.SpecifiedSparsePhiConfig_Mode_SparseTopics
        self._sparse_by_columns = True
      else:
        config.mode = library.SpecifiedSparsePhiConfig_Mode_SparseTokens
        self._sparse_by_columns = False

    self._name = name
    self._tau = tau
    self._type = PhiRegularizer_Type
    self._config = config
    self._type = library.RegularizerConfig_Type_SpecifiedSparsePhi
    self._regularizer = None  # reserve place for regularizer
    
  @property
  def name(self): return self._name
  @property
  def tau(self): return self._tau 
  @property
  def type(self): return self._type   
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
  def regularizer(self): return self._regularizer

  @regularizer.setter
  def regularizer(self, regularizer): self._regularizer = regularizer

  @tau.setter
  def tau(self, tau): self._tau = tau

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
    if sparse_by_columns == True:
      config.mode = library.SpecifiedSparsePhiConfig_Mode_SparseTopics
    else:
      config.mode = library.SpecifiedSparsePhiConfig_Mode_SparseTokens
    self.regularizer.Reconfigure(self.regularizer.config_.type, config)

#######################################################################################################################
class ImproveCoherencePhiRegularizer(object):
  """ ImproveCoherencePhiRegularizer is a regularizer in ArtmModel (public class).
  Args:
  - name --- the identifier of regularizer. Is string, default = None
  - tau --- the coefficient of regularization for this regularizer, double, default = 1.0
  - class_ids --- list of class_ids to regularize. Is list of strings, default = None
  - topic_names --- list of names of topics to regularize. Is list of strings, default = None
  - dictionary_name --- BigARTM collection dictionary. Is string, default = None
  """
  def __init__(self, name=None, tau=1.0, class_ids=None, topic_names=None, dictionary_name=None):
    config = messages_pb2.ImproveCoherencePhiConfig()
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
    self._type = PhiRegularizer_Type
    self._config = config
    self._type = library.RegularizerConfig_Type_ImproveCoherencePhi
    self._regularizer = None  # reserve place for regularizer
    
  @property
  def name(self): return self._name
  @property
  def tau(self): return self._tau
  @property
  def type(self): return self._type 
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
  def regularizer(self): return self._regularizer

  @regularizer.setter
  def regularizer(self, regularizer): self._regularizer = regularizer

  @tau.setter
  def tau(self, tau): self._tau = tau

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

#######################################################################################################################
# SECTION OF SCORE CLASSES
#######################################################################################################################
class SparsityPhiScore(object):
  """ SparsityPhiScore is a score in ArtmModel (public class).
  Args:
  - name --- the identifier of score. Is string, default = None
  - class_id --- class_id to score. Is string, default = None
  - topic_names --- list of names of topics to score. Is list of strings, default = None
  - eps --- the tolerance const, everything < eps considered to be zero. Is double, default = None
  """
  def __init__(self, name=None, class_id=None, topic_names=None, eps=None):
    config = messages_pb2.SparsityPhiScoreConfig()
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
      config.eps = eps
      self._eps = eps

    self._name = name
    self._config = config
    self._type = library.ScoreConfig_Type_SparsityPhi
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

#######################################################################################################################
class SparsityThetaScore(object):
  """ SparsityThetaScore is a score in ArtmModel (public class).
  Args:
  - name --- the identifier of score. Is string, default = None
  - topic_names --- list of names of topics to score. Is list of strings, default = None
  - eps --- the tolerance const, everything < eps considered to be zero. Is double, default = None
  """
  def __init__(self, name=None, topic_names=None, eps=None):
    config = messages_pb2.SparsityThetaScoreConfig()
    self._topic_names = []
    self._eps = 1e-37

    if name is None:
      name = "SparsityThetaScore:" + uuid.uuid1().urn
    if not topic_names is None:
      config.ClearField('topic_name')
      for topic_name in topic_names:
        config.topic_name.append(topic_name)
        self._topic_names.append(topic_name)
    if not eps is None:
      config.eps = eps
      self._eps = eps

    self._name = name
    self._config = config
    self._type = library.ScoreConfig_Type_SparsityTheta
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

#######################################################################################################################
class PerplexityScore(object):
  """ PerplexityScore is a score in ArtmModel (public class).
  Args:
  - name --- the identifier of score. Is string, default = None
  - class_id --- class_id to score. Is string, default = None
  - topic_names --- list of names of topics to score Theta sparsity. Is list of strings, default = None
  - eps --- the tolerance const for Theta sparsity, everything < eps considered to be zero. Is double, default = None
  - dictionary_name --- BigARTM collection dictionary. Is string, default = None
  - use_unigram_document_model --- use uni-gram document/collection model if token's counter == 0. 
                                   Is bool, default = None
  """
  def __init__(self, name=None, class_id=None, topic_names=None, eps=None,
               dictionary_name=None, use_unigram_document_model=None):
    config = messages_pb2.PerplexityScoreConfig()
    self._class_id = '@default_class'
    self._topic_names = []
    self._eps = 1e-37
    self._dictionary_name = ''
    self._use_unigram_document_model = True

    if name is None:
      name = "PerplexityScore:" + uuid.uuid1().urn
    if not class_id is None:
      config.class_id = class_id
      self._class_id = class_id
    if not topic_names is None:
      config.ClearField('theta_sparsity_topic_name')
      for topic_name in topic_names:
        config.theta_sparsity_topic_name.append(topic_name)
        self._topic_names.append(topic_name)
    if not eps is None:
      config.theta_sparsity_eps = eps
      self._eps = eps
    if not dictionary_name is None:
      self._dictionary_name = dictionary_name
      config.dictionary_name = dictionary_name
    if not use_unigram_document_model is None:
      self._use_unigram_document_model = use_unigram_document_model
      if use_unigram_document_model == True:
        config.model_type = library.PerplexityScoreConfig_Type_UnigramDocumentModel
      else:
        config.model_type = library.PerplexityScoreConfig_Type_UnigramCollectionModel

    self._name = name
    self._config = config
    self._type = library.ScoreConfig_Type_Perplexity
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
  def dictionary_name(self): return self._dictionary_name
  @property
  def use_unigram_document_model(self): return self._use_unigram_document_model
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
    if use_unigram_document_model == True:
      score_config.model_type = library.PerplexityScoreConfig_Type_UnigramDocumentModel
    else:
      score_config.model_type = library.PerplexityScoreConfig_Type_UnigramCollectionModel
    reconfigure_score_in_master(self._master, score_config, self._name)

#######################################################################################################################
class ItemsProcessedScore(object):
  """ ItemsProcessedScore is a score in ArtmModel (public class).
  Args:
  - name --- the identifier of score. Is string, default = None
  """
  def __init__(self, name=None):
    config = messages_pb2.ItemsProcessedScoreConfig()

    if name is None:
      name = "PerplexityScore:" + uuid.uuid1().urn

    self._name = name
    self._config = config
    self._type = library.ScoreConfig_Type_ItemsProcessed
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

#######################################################################################################################
class TopTokensScore(object):
  """ TopTokensScore is a score in ArtmModel (public class).
  Args:
  - name --- the identifier of score. Is string, default = None
  - class_id --- class_id to score. Is string, default = None
  - topic_names --- list of names of topics to score Theta sparsity. Is list of strings, default = None
  - num_tokens --- Number of tokens with max probability in each topic. Is int, default = None
  - dictionary_name --- BigARTM collection dictionary. Is string, default = None
  """
  def __init__(self, name=None, class_id=None, topic_names=None, num_tokens=None, dictionary_name=None):
    config = messages_pb2.TopTokensScoreConfig()
    self._class_id = '@default_class'
    self._topic_names = []
    self._num_tokens = 10
    self._dictionary_name = ''

    if name is None:
      name = "TopTokensScore:" + uuid.uuid1().urn
    if not class_id is None:
      config.class_id = class_id
      self._class_id = class_id
    if not topic_names is None:
      config.ClearField('theta_sparsity_topic_name')
      for topic_name in topic_names:
        config.theta_sparsity_topic_name.append(topic_name)
        self._topic_names.append(topic_name)
    if not num_tokens is None:
      config.num_tokens = num_tokens
      self._num_tokens = num_tokens
    if not dictionary_name is None:
      self._dictionary_name = dictionary_name
      config.cooccurrence_dictionary_name = dictionary_name

    self._name = name
    self._config = config
    self._type = library.ScoreConfig_Type_TopTokens
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
  def num_tokens(self): return self._num_tokens
  @property
  def dictionary_name(self): return self._dictionary_name
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

#######################################################################################################################
class ThetaSnippetScore(object):
  """ ThetaSnippetScore is a score in ArtmModel (public class).
  Args:
  - name --- the identifier of score. Is string, default = None
  - item_ids --- list of names of items to show. Is list of ints, default = None
  - num_items --- number of theta vectors to show from the beginning (no sense if item_ids given).
                  Is int, default = None
  """
  def __init__(self, name=None, item_ids=None, num_items=None):
    config = messages_pb2.ThetaSnippetScoreConfig()
    self._item_ids = []
    self._num_items = 10

    if name is None:
      name = "ThetaSnippetScore:" + uuid.uuid1().urn
    if not item_ids is None:
      config.ClearField('item_id')
      for item_id in item_ids:
        config.item_id.append(item_id)
        self._item_ids.append(item_id)
    if not num_items is None:
      config.item_count = num_items
      self._num_items = num_items

    self._name = name
    self._config = config
    self._type = library.ScoreConfig_Type_ThetaSnippet
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
  def item_ids(self): return self._item_ids
  @property
  def num_items(self): return self._num_items
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

#######################################################################################################################
class TopicKernelScore(object):
  """ TopicKernelScore is a score in ArtmModel (public class).
  Args:
  - name --- the identifier of score. Is string, default = None
  - class_id --- class_id to score. Is string, default = None
  - topic_names --- list of names of topics to score Theta sparsity. Is list of strings, default = None
  - eps --- the tolerance const for counting, everything < eps considered to be zero. Is double, default = None
  - dictionary_name --- BigARTM collection dictionary_name. Is string, default = None
  - probability_mass_threshold --- the threshold for p(t|w) values to get token into topic kernel.
                                   Is double, in (0,1), default = None
  """
  def __init__(self, name=None, class_id=None, topic_names=None, eps=None,
               dictionary_name=None, probability_mass_threshold=None):
    config = messages_pb2.TopicKernelScoreConfig()
    self._class_id = '@default_class'
    self._topic_names = []
    self._eps = 1e-37
    self._dictionary_name = ''
    self._probability_mass_threshold = 0.1

    if name is None:
      name = "TopicKernelScore:" + uuid.uuid1().urn
    if not class_id is None:
      config.class_id = class_id
      self._class_id = class_id
    if not topic_names is None:
      config.ClearField('theta_sparsity_topic_name')
      for topic_name in topic_names:
        config.theta_sparsity_topic_name.append(topic_name)
        self._topic_names.append(topic_name)
    if not eps is None:
      config.theta_sparsity_eps = eps
      self._eps = eps
    if not dictionary_name is None:
      self._dictionary_name = dictionary_name
      config.dictionary_name = dictionary_name
    if not probability_mass_threshold is None:
      config.probability_mass_threshold = probability_mass_threshold
      self._probability_mass_threshold = probability_mass_threshold

    self._name = name
    self._config = config
    self._type = library.ScoreConfig_Type_TopicKernel
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
  def dictionary_name(self): return self._dictionary_name
  @property
  def probability_mass_threshold(self): return self._probability_mass_threshold
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

  def add(self, score=None):
    """ SparsityPhiScoreInfo.add() --- add info about score after synchronization.
    Args:
    - score --- reference to score object, default = None (means "Add None values")
    """
    if not score is None:
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
  def name(self): return self._name
  @property
  def value(self):
    """ value of Phi sparsity on synchronizations. Is list of scalars """  
    return self._value  
  @property
  def zero_tokens(self):
    """ number of zero rows in Phi on synchronizations. Is list of scalars """  
    return self._zero_tokens
  @property
  def total_tokens(self):
    """ total number of rows in Phi on synchronizations. Is list of scalars """ 
    return self._total_tokens
 
#######################################################################################################################
class SparsityThetaScoreInfo(object):
  """ SparsityThetaScoreInfo represents a result of counting SparsityThetaScore (private class).
  Args:
  - score --- reference to score object, no default
  """
  def __init__(self, score):   
    self._name = score.name
    self._value = []
    self._zero_topics = []
    self._total_topics = []

  def add(self, score=None):
    """ SparsityThetaScoreInfo.add() --- add info about score after synchronization.
    Args:
    - score --- reference to score object, default = None (means "Add None values")
    """
    if not score is None:
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
  def name(self): return self._name
  @property
  def value(self):
    """ value of Theta sparsity on synchronizations. Is list of scalars """  
    return self._value  
  @property
  def zero_topics(self):
    """ number of zero rows in Theta on synchronizations. Is list of scalars """  
    return self._zero_topics
  @property
  def total_topics(self):
    """ total number of rows in Theta on synchronizations. Is list of scalars """ 
    return self._total_topics
 
#######################################################################################################################
class PerplexityScoreInfo(object):
  """ PerplexityScoreInfo represents a result of counting PerplexityScore (private class).
  Args:
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
    """ PerplexityScoreInfo.add() --- add info about score after synchronization.
    Args:
    - score --- reference to score object, default = None (means "Add None values")
    """
    if not score is None:
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
  def name(self): return self._name
  @property
  def value(self):
    """ value of perplexity on synchronizations. Is list of scalars """  
    return self._value  
  @property
  def raw(self):
    """ raw value in formula of perplexity on synchronizations. Is list of scalars """  
    return self._raw
  @property  
  def normalizer(self):
    """ normalizer value in formula of perplexity on synchronizations. Is list of scalars """  
    return self._normalizer  
  @property
  def zero_tokens(self):
    """ number of tokens with zero counters on synchronizations. Is list of scalars """  
    return self._zero_tokens
  @property
  def theta_sparsity_value(self):
    """ Theta sparsity value on synchronizations. Is list of scalars """ 
    return self._theta_sparsity_value
  @property
  def theta_sparsity_zero_topics(self):
    """ number of zero rows in Theta on synchronizations. Is list of scalars """ 
    return self._theta_sparsity_zero_topics 
  @property
  def theta_sparsity_total_topics(self):
    """ total number of rows in Theta on synchronizations. Is list of scalars """ 
    return self._theta_sparsity_total_topics

#######################################################################################################################
class ItemsProcessedScoreInfo(object):
  """ ItemsProcessedScoreInfo represents a result of counting ItemsProcessedScore (private class).
  Args:
  - score --- reference to score object, no default
  """
  def __init__(self, score):
    self._name = score.name
    self._value = []

  def add(self, score=None):
    """ ItemsProcessedScoreInfo.add() --- add info about score after synchronization.
    Args:
    - score --- reference to score object, default = None (means "Add None values")
    """
    if not score is None:
      _data = messages_pb2.ItemsProcessedScore()
      _data = score.score.GetValue(score._model)
      self._value.append(_data.value)
    else:
      self._value.append(None)

  @property
  def name(self): return self._name
  @property
  def value(self):
    """ total number of processed documents on synchronizations. Is list of scalars """  
    return self._value  

#######################################################################################################################
class TopTokensScoreInfo(object):
  """ TopTokensScoreInfo represents a result of counting TopTokensScore (private class).
  Args:
  - score --- reference to score object, no default
  """
  def __init__(self, score):
    self._name = score.name
    self._num_tokens = []
    self._topic_info = []
    self._average_coherence = []

  def add(self, score=None):
    """ TopTokensScoreInfo.add() --- add info about score after synchronization.
    Args:
    - score --- reference to score object, default = None (means "Add None values")
    """
    if not score is None:
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
        self._topic_info[index][topic_name] = namedtuple('TopTokensScoreTuple', ['tokens', 'weights', 'coherence'])
        self._topic_info[index][topic_name].tokens = tokens
        self._topic_info[index][topic_name].weights = weights
        self._topic_info[index][topic_name].coherence = coherence

      self._average_coherence. append(_data.average_coherence)
    else:
      self._num_tokens.append(None)
      self._topic_info.append(None)
      self._average_coherence.append(None)

  @property
  def name(self): return self._name
  @property
  def num_tokens(self):
    """ reqested number of top tokens in each topic on synchronizations. Is list of scalars """  
    return self._num_tokens

  @property
  def topic_info(self):
    """ information about top tokens per topic on synchronizations. Is list of sets. Set contains 
        information about topics, key --- name of topic, value --- named tuple:
        - *.topic_info[sync_index][topic_name].tokens --- list of top tokens for this topic.
        - *.topic_info[sync_index][topic_name].weights --- list of weights (probabilities), corresponds the tokens.
        - *.topic_info[sync_index][topic_name].coherence --- the coherency of topic due to it's top tokens.
    """  
    return self._topic_info

  @property
  def average_coherence(self):
    """ average coherence of top tokens in all requested topics on synchronizations. Is list of scalars """  
    return self._average_coherence

#######################################################################################################################
class TopicKernelScoreInfo(object):
  """ TopicKernelScoreInfo represents a result of counting TopicKernelScore (private class).
  Args:
  - score --- reference to score object, no default
  """
  def __init__(self, score):
    self._name = score.name
    self._topic_info = []
    self._average_coherence = []
    self._average_kernel_size = []
    self._average_kernel_contrast = []
    self._average_kernel_purity = []

  def add(self, score=None):
    """ TopicKernelScoreInfo.add() --- add info about score after synchronization.
    Args:
    - score --- reference to score object, default = None (means "Add None values")
    """
    if not score is None:
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
        self._topic_info[index][topic_name] = namedtuple('TopicKernelScoreTuple',
                                                         ['tokens', 'size', 'contrast', 'purity', 'coherence'])
        self._topic_info[index][topic_name].tokens = tokens
        self._topic_info[index][topic_name].size = _data.kernel_size.value[topic_index]
        self._topic_info[index][topic_name].contrast = _data.kernel_purity.value[topic_index]
        self._topic_info[index][topic_name].purity = _data.kernel_contrast.value[topic_index]
        self._topic_info[index][topic_name].coherence = coherence

      self._average_coherence.append(_data.average_coherence)
      self._average_kernel_size.append(_data.average_kernel_size)
      self._average_kernel_contrast.append(_data.average_kernel_contrast)
      self._average_kernel_purity.append(_data.average_kernel_purity)
    else:
      self._topic_info.append(None)
      self._average_coherence.append(None)
      self._average_kernel_size.append(None)
      self._average_kernel_contrast.append(None)
      self._average_kernel_purity.append(None)

  @property
  def name(self): return self._name

  @property
  def topic_info(self):
    """ information about kernel tokens per topic on synchronizations. Is list of sets. Set contains 
        information about topics, key --- name of topic, value --- named tuple:
        - *.topic_info[sync_index][topic_name].tokens --- list of kernel tokens for this topic.
        - *.topic_info[sync_index][topic_name].size --- size of kernel for this topic.
        - *.topic_info[sync_index][topic_name].contrast --- contrast of kernel for this topic.
        - *.topic_info[sync_index][topic_name].purity --- purity of kernel for this topic.
        - *.topic_info[sync_index][topic_name].coherence --- the coherency of topic due to it's kernel.
    """  
    return self._topic_info

  @property
  def average_coherence(self):
    """ average coherence of kernel tokens in all requested topics on synchronizations. Is list of scalars """  
    return self._average_coherence
  @property
  def average_kernel_size(self):
    """ average kernel size of all requested topics on synchronizations. Is list of scalars """  
    return self._average_kernel_size
  @property
  def average_kernel_contrast(self):
    """ average kernel contrast of all requested topics on synchronizations. Is list of scalars """  
    return self._average_kernel_contrast
  @property
  def average_kernel_purity(self):
    """ average kernel purity of all requested topics on synchronizations. Is list of scalars """  
    return self._average_kernel_purity

#######################################################################################################################
class ThetaSnippetScoreInfo(object):
  """ ThetaSnippetScoreInfo represents a result of counting ThetaSnippetScore (private class).
  Args:
  - score --- reference to score object, no default
  """
  def __init__(self, score):
    self._name = score.name
    self._document_ids = []
    self._snippet = []

  def add(self, score=None):
    """ ThetaSnippetScoreInfo.add() --- add info about score after synchronization.
    Args:
    - score --- reference to score object, default = None (means "Add None values")
    """
    if not score is None:
      _data = messages_pb2.ThetaSnippetScore()
      _data = score.score.GetValue(score._model)

      self._document_ids .append([item_id for item_id in _data.item_id])
      self._snippet.append([[theta_td for theta_td in theta_d.value] for theta_d in _data.values])
    else:
      self._document_ids.append(None)
      self._snippet.append(None)

  @property
  def name(self): return self._name

  @property
  def snippet(self):
    """ the snippet (part) of Theta corresponds to documents from document_ids. Is list of lists of scalars, 
        each internal list --- theta_d vector for document d, in direct order of document_ids
    """  
    return self._snippet

  @property
  def document_ids(self):
    """ ids of documents in snippet on synchronizations. Is list of scalars """  
    return self._document_ids

#######################################################################################################################