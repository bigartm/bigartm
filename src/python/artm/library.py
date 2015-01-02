# Copyright 2014, Additive Regularization of Topic Models.

import messages_pb2
import sys
import os
import ctypes
import uuid
import random
import glob
from ctypes import *

#################################################################################

ARTM_SUCCESS = 0                   # Has no corresponding exception type.
ARTM_STILL_WORKING = -1            # Has no corresponding exception type.
ARTM_INTERNAL_ERROR = -2
ARTM_ARGUMENT_OUT_OF_RANGE = -3
ARTM_INVALID_MASTER_ID = -4
ARTM_CORRUPTED_MESSAGE = -5
ARTM_INVALID_OPERATION = -6
ARTM_DISK_READ_ERROR = -7
ARTM_DISK_WRITE_ERROR = -8
ARTM_NETWORK_ERROR = -9

Stream_Type_Global = 0
Stream_Type_ItemIdModulus = 1
RegularizerConfig_Type_SmoothSparseTheta = 0
RegularizerConfig_Type_SmoothSparsePhi = 1
RegularizerConfig_Type_DecorrelatorPhi = 2
ScoreConfig_Type_Perplexity = 0
ScoreData_Type_Perplexity = 0
ScoreConfig_Type_SparsityTheta = 1
ScoreData_Type_SparsityTheta = 1
ScoreConfig_Type_SparsityPhi = 2
ScoreData_Type_SparsityPhi = 2
ScoreConfig_Type_ItemsProcessed = 3
ScoreData_Type_ItemsProcessed = 3
ScoreConfig_Type_TopTokens = 4
ScoreData_Type_TopTokens = 4
ScoreConfig_Type_ThetaSnippet = 5
ScoreData_Type_ThetaSnippet = 5
ScoreConfig_Type_TopicKernel = 6
ScoreData_Type_TopicKernel = 6
PerplexityScoreConfig_Type_UnigramDocumentModel = 0
PerplexityScoreConfig_Type_UnigramCollectionModel = 1
CollectionParserConfig_Format_BagOfWordsUci = 0
MasterComponentConfig_ModusOperandi_Local = 0
MasterComponentConfig_ModusOperandi_Network = 1

#################################################################################

class InternalError(BaseException) : pass
class ArgumentOutOfRangeException(BaseException) : pass
class InvalidMasterIdException(BaseException) : pass
class CorruptedMessageException(BaseException) : pass
class InvalidOperationException(BaseException) : pass
class DiskReadException(BaseException) : pass
class DiskWriteException(BaseException) : pass
class NetworkException(BaseException) : pass


def GetLastErrorMessage(lib):
  error_message = lib.ArtmGetLastErrorMessage()
  return ctypes.c_char_p(error_message).value


def HandleErrorCode(lib, artm_error_code):
  if (artm_error_code == ARTM_SUCCESS) or (artm_error_code == ARTM_STILL_WORKING) or (artm_error_code >= 0):
    return artm_error_code
  elif artm_error_code == ARTM_INTERNAL_ERROR:
    raise InternalError(GetLastErrorMessage(lib))
  elif artm_error_code == ARTM_ARGUMENT_OUT_OF_RANGE:
    raise ArgumentOutOfRangeException(GetLastErrorMessage(lib))
  elif artm_error_code == ARTM_INVALID_MASTER_ID:
      raise InvalidMasterIdException(GetLastErrorMessage(lib))
  elif artm_error_code == ARTM_CORRUPTED_MESSAGE:
      raise CorruptedMessageException(GetLastErrorMessage(lib))
  elif artm_error_code == ARTM_INVALID_OPERATION:
      raise InvalidOperationException(GetLastErrorMessage(lib))
  elif artm_error_code == ARTM_DISK_READ_ERROR:
      raise DiskReadException(GetLastErrorMessage(lib))
  elif artm_error_code == ARTM_DISK_WRITE_ERROR:
      raise DiskWriteException(GetLastErrorMessage(lib))
  elif artm_error_code == ARTM_NETWORK_ERROR:
      raise NetworkException(GetLastErrorMessage(lib))
  else:
    raise InternalError("Unknown error code: " + str(artm_error_code))

#################################################################################

class Library:
  def __init__(self, artm_shared_library = ""):
    if not artm_shared_library:
      if sys.platform.count('linux') == 1:
        artm_shared_library = 'libartm.so'
      else:
        artm_shared_library = 'artm.dll'

    try:
      self.lib_ = ctypes.CDLL(artm_shared_library)
      return
    except OSError as e:
      print str(e) + ", fall back to ARTM_SHARED_LIBRARY environment variable"

    self.lib_ = ctypes.CDLL(os.environ['ARTM_SHARED_LIBRARY'])

  def CreateMasterComponent(self, config = messages_pb2.MasterComponentConfig()):
    return MasterComponent(config, self.lib_)

  def CreateNodeController(self, endpoint):
    return NodeController(endpoint, self.lib_)

  def SaveBatch(self, batch, disk_path):
    batch_blob = batch.SerializeToString()
    batch_blob_p = ctypes.create_string_buffer(batch_blob)
    disk_path_p = ctypes.create_string_buffer(disk_path)
    HandleErrorCode(self.lib_, self.lib_.ArtmSaveBatch(disk_path_p, len(batch_blob), batch_blob_p))

  def ParseCollection(self, collection_parser_config):
    config_blob = collection_parser_config.SerializeToString()
    config_blob_p = ctypes.create_string_buffer(config_blob)
    length = HandleErrorCode(self.lib_, self.lib_.ArtmRequestParseCollection(
                             len(config_blob), config_blob_p))

    dictionary_blob = ctypes.create_string_buffer(length)
    HandleErrorCode(self.lib_, self.lib_.ArtmCopyRequestResult(length, dictionary_blob))

    dictionary = messages_pb2.DictionaryConfig()
    dictionary.ParseFromString(dictionary_blob)
    return dictionary

  def LoadDictionary(self, full_filename):
    full_filename_p = ctypes.create_string_buffer(full_filename)
    length = HandleErrorCode(self.lib_, self.lib_.ArtmRequestLoadDictionary(full_filename_p))

    dictionary_blob = ctypes.create_string_buffer(length)
    HandleErrorCode(self.lib_, self.lib_.ArtmCopyRequestResult(length, dictionary_blob))

    dictionary = messages_pb2.DictionaryConfig()
    dictionary.ParseFromString(dictionary_blob)
    return dictionary

  def LoadBatch(self, full_filename):
    full_filename_p = ctypes.create_string_buffer(full_filename)
    length = HandleErrorCode(self.lib_, self.lib_.ArtmRequestLoadBatch(full_filename_p))

    message_blob = ctypes.create_string_buffer(length)
    HandleErrorCode(self.lib_, self.lib_.ArtmCopyRequestResult(length, message_blob))

    batch = messages_pb2.Batch()
    batch.ParseFromString(message_blob)
    return batch

  def ParseCollectionOrLoadDictionary(self, docword_file_path, vocab_file_path, target_folder):
    batches_found = len(glob.glob(target_folder + "/*.batch"))
    if batches_found == 0:
      print "No batches found, parsing them from textual collection...",
      collection_parser_config = messages_pb2.CollectionParserConfig();
      collection_parser_config.format = CollectionParserConfig_Format_BagOfWordsUci

      collection_parser_config.docword_file_path = docword_file_path
      collection_parser_config.vocab_file_path = vocab_file_path
      collection_parser_config.target_folder = target_folder
      collection_parser_config.dictionary_file_name = 'dictionary'
      unique_tokens = self.ParseCollection(collection_parser_config);
      print " OK."
      return unique_tokens
    else:
      print "Found " + str(batches_found) + " batches, using them."
      return self.LoadDictionary(target_folder + '/dictionary');

#################################################################################

class MasterComponent:
  def __init__(self, config = messages_pb2.MasterComponentConfig(), lib = None, disk_path = None, proxy_endpoint = None):
    if (lib is None):
      lib = Library().lib_

    if (disk_path is not None):
      config.disk_path = disk_path

    self.lib_ = lib
    master_config_blob = config.SerializeToString()
    master_config_blob_p = ctypes.create_string_buffer(master_config_blob)

    if (isinstance(config, messages_pb2.MasterComponentConfig)):
      self.config_ = config
      self.id_ = HandleErrorCode(self.lib_, self.lib_.ArtmCreateMasterComponent(
                 len(master_config_blob), master_config_blob_p))
      return

    if (isinstance(config, messages_pb2.MasterProxyConfig)):
      self.config_ = config.config
      self.id_ = HandleErrorCode(self.lib_, self.lib_.ArtmCreateMasterProxy(
                 len(master_config_blob), master_config_blob_p))
      return

    raise InvalidOperation(GetLastErrorMessage(self.lib_))

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    self.Dispose()

  def Dispose(self):
    self.lib_.ArtmDisposeMasterComponent(self.id_)
    self.id_ = -1

  def config(self):
    return self.config_

  def CreateModel(self, config = messages_pb2.ModelConfig(), 
                  topics_count = None, inner_iterations_count = None):
    if (topics_count is not None):
      config.topics_count = topics_count
    if (inner_iterations_count is not None):
      config.inner_iterations_count = inner_iterations_count
    return Model(self, config)

  def RemoveModel(self, model):
    model.__Dispose__()

  def CreateRegularizer(self, name, type, config):
    general_config = messages_pb2.RegularizerConfig()
    general_config.name = name
    general_config.type = type
    general_config.config = config.SerializeToString()
    return Regularizer(self, general_config)

  def RemoveRegularizer(self, regularizer):
    regularizer.__Dispose__()

  def CreateSmoothSparseThetaRegularizer(self, name = None, config = messages_pb2.SmoothSparseThetaConfig()):
    if (name is None):
      name = "SmoothSparseThetaRegularizer:" + uuid.uuid1().urn
    return self.CreateRegularizer(name, RegularizerConfig_Type_SmoothSparseTheta, config)

  def CreateSmoothSparsePhiRegularizer(self, name = None, config = messages_pb2.SmoothSparsePhiConfig()):
    if (name is None):
      name = "SmoothSparsePhiRegularizer:" + uuid.uuid1().urn
    return self.CreateRegularizer(name, RegularizerConfig_Type_SmoothSparsePhi, config)

  def CreateDecorrelatorPhiRegularizer(self, name = None, config = messages_pb2.DecorrelatorPhiConfig()):
    if (name is None):
      name = "DecorrelatorPhiRegularizer:" + uuid.uuid1().urn
    return self.CreateRegularizer(name, RegularizerConfig_Type_DecorrelatorPhi, config)

  def CreateScore(self, name, type, config):
    master_config = messages_pb2.MasterComponentConfig();
    master_config.CopyFrom(self.config_);
    score_config = master_config.score_config.add();
    score_config.name = name
    score_config.type = type;
    score_config.config = config.SerializeToString();
    self.Reconfigure(master_config)
    return Score(self, name)

  def CreatePerplexityScore(self, name = None, config = messages_pb2.PerplexityScoreConfig(), stream_name = None):
    if (name is None):
      name = "PerplexityScore:" + uuid.uuid1().urn
    if (stream_name is not None):
      config.stream_name = stream_name
    return self.CreateScore(name, ScoreConfig_Type_Perplexity, config)

  def CreateSparsityThetaScore(self, name = None, config = messages_pb2.SparsityThetaScoreConfig()):
    if (name is None):
      name = "SparsityThetaScore:" + uuid.uuid1().urn
    return self.CreateScore(name, ScoreConfig_Type_SparsityTheta, config)

  def CreateSparsityPhiScore(self, name = None, config = messages_pb2.SparsityPhiScoreConfig()):
    if (name is None):
      name = "SparsityPhiScore:" + uuid.uuid1().urn
    return self.CreateScore(name, ScoreConfig_Type_SparsityPhi, config)

  def CreateItemsProcessedScore(self, name = None, config = messages_pb2.ItemsProcessedScoreConfig()):
    if (name is None):
      name = "ItemsProcessedScore:" + uuid.uuid1().urn
    return self.CreateScore(name, ScoreConfig_Type_ItemsProcessed, config)

  def CreateTopTokensScore(self, name = None, config = messages_pb2.TopTokensScoreConfig(), num_tokens = None):
    if (name is None):
      name = "TopTokensScore:" + uuid.uuid1().urn
    if (num_tokens is not None):
      config.num_tokens = num_tokens
    return self.CreateScore(name, ScoreConfig_Type_TopTokens, config)

  def CreateThetaSnippetScore(self, name = None, config = messages_pb2.ThetaSnippetScoreConfig()):
    if (name is None):
      name = "ThetaSnippetScore:" + uuid.uuid1().urn
    if (len(config.item_id) == 0):
      for i in range(1, 11): config.item_id.append(i)
    return self.CreateScore(name, ScoreConfig_Type_ThetaSnippet, config)

  def CreateTopicKernelScore(self, name = None, config = messages_pb2.TopicKernelScoreConfig()):
    if (name is None):
      name = "TopicKernelScore:" + uuid.uuid1().urn
    return self.CreateScore(name, ScoreConfig_Type_TopicKernel, config)

  def RemoveScore(self, name):
    raise NotImplementedError

  def CreateDictionary(self, config):
    return Dictionary(self, config)

  def RemoveDictionary(self, dictionary):
    dictionary.__Dispose__()

  def Reconfigure(self, config = None):
    if (config is None):
      config = self.config_
    config_blob = config.SerializeToString()
    config_blob_p = ctypes.create_string_buffer(config_blob)
    HandleErrorCode(self.lib_, self.lib_.ArtmReconfigureMasterComponent(self.id_, len(config_blob), config_blob_p))
    self.config_.CopyFrom(config)

  def AddBatch(self, batch):
    batch_blob = batch.SerializeToString()
    batch_blob_p = ctypes.create_string_buffer(batch_blob)
    HandleErrorCode(self.lib_, self.lib_.ArtmAddBatch(self.id_, len(batch_blob), batch_blob_p))

  def InvokeIteration(self, iterations_count = 1):
    HandleErrorCode(self.lib_, self.lib_.ArtmInvokeIteration(self.id_, iterations_count))

  def WaitIdle(self, timeout = -1):
    result = self.lib_.ArtmWaitIdle(self.id_, timeout)
    result = HandleErrorCode(self.lib_, result)
    return False if (result == ARTM_STILL_WORKING) else True

  def CreateStream(self, stream):
    s = self.config_.stream.add()
    s.CopyFrom(stream)
    self.Reconfigure(self.config_)

  def RemoveStream(self, stream_name):
    new_config_ = messages_pb2.MasterComponentConfig()
    new_config_.CopyFrom(self.config_)
    new_config_.ClearField('stream')

    for stream_index in range(0, len(self.config_.stream)):
      if (self.config_.stream[stream_index].name != stream_name):
        s = new_config_.stream.add()
        s.CopyFrom(self.config_.stream[stream_index])
    self.Reconfigure(new_config_)

  def GetTopicModel(self, model = None, args = messages_pb2.GetTopicModelArgs()):
    if (model is not None):
      args.model_name = model.name()
    
    args_blob = args.SerializeToString()
    length = HandleErrorCode(self.lib_, self.lib_.ArtmRequestTopicModel(self.id_, len(args_blob), args_blob))

    topic_model_blob = ctypes.create_string_buffer(length)
    HandleErrorCode(self.lib_, self.lib_.ArtmCopyRequestResult(length, topic_model_blob))

    topic_model = messages_pb2.TopicModel()
    topic_model.ParseFromString(topic_model_blob)
    return topic_model

  def GetRegularizerState(self, regularizer_name):
    length = HandleErrorCode(self.lib_, self.lib_.ArtmRequestRegularizerState(self.id_, regularizer_name))

    state_blob = ctypes.create_string_buffer(length)
    HandleErrorCode(self.lib_, self.lib_.ArtmCopyRequestResult(length, state_blob))

    regularizer_state = messages_pb2.RegularizerInternalState()
    regularizer_state.ParseFromString(state_blob)
    return regularizer_state

  def GetThetaMatrix(self, model = None, batch = None, args = messages_pb2.GetThetaMatrixArgs()):
    if (model is not None):
      args.model_name = model.name()
    if (batch is not None):
      args.batch.CopyFrom(batch)
    args_blob = args.SerializeToString()
    length = HandleErrorCode(self.lib_,  self.lib_.ArtmRequestThetaMatrix(self.id_, len(args_blob), args_blob))
    blob = ctypes.create_string_buffer(length)
    HandleErrorCode(self.lib_, self.lib_.ArtmCopyRequestResult(length, blob))

    theta_matrix = messages_pb2.ThetaMatrix()
    theta_matrix.ParseFromString(blob)
    return theta_matrix

#################################################################################

class NodeController:
  def __init__(self, endpoint, lib = None):
    config = messages_pb2.NodeControllerConfig();
    config.create_endpoint = endpoint;

    if (lib is None):
      lib = Library().lib_

    self.lib_ = lib
    config_blob = config.SerializeToString()
    config_blob_p = ctypes.create_string_buffer(config_blob)

    self.id_ = HandleErrorCode(self.lib_, self.lib_.ArtmCreateNodeController(
                               len(config_blob), config_blob_p))

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    self.Dispose()

  def Dispose(self):
    self.lib_.ArtmDisposeNodeController(self.id_)
    self.id_ = -1

#################################################################################

class Model:
  def __init__(self, master_component, config):
    self.lib_ = master_component.lib_
    self.master_id_ = master_component.id_
    self.master_component = master_component
    self.config_ = config
    self.config_.name = uuid.uuid1().urn
    model_config_blob = config.SerializeToString()
    model_config_blob_p = ctypes.create_string_buffer(model_config_blob)
    HandleErrorCode(self.lib_, self.lib_.ArtmCreateModel(self.master_id_,
                    len(model_config_blob), model_config_blob_p))

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    __Dispose__(self)

  def __Dispose__(self):
    self.lib_.ArtmDisposeModel(self.master_id_, self.config_.name)
    self.config_.name = ''
    self.master_id_ = -1

  def name(self):
    return self.config_.name

  def topics_count(self):
    return self.config_.topics_count

  def config(self):
    return self.config_

  def Reconfigure(self, config = None):
    if (config is None):
      config = self.config_

    model_config_blob = config.SerializeToString()
    model_config_blob_p = ctypes.create_string_buffer(model_config_blob)
    HandleErrorCode(self.lib_, self.lib_.ArtmReconfigureModel(self.master_id_,
                    len(model_config_blob), model_config_blob_p))
    self.config_.CopyFrom(config)

  def Synchronize(self, decay_weight = 0.0, apply_weight = 1.0, invoke_regularizers = True):
    args = messages_pb2.SynchronizeModelArgs();
    args.model_name = self.name()
    args.decay_weight = decay_weight
    args.apply_weight = apply_weight
    args.invoke_regularizers = invoke_regularizers
    args_blob = args.SerializeToString()
    args_blob_p = ctypes.create_string_buffer(args_blob)
    HandleErrorCode(self.lib_, self.lib_.ArtmSynchronizeModel(
                     self.master_id_, len(args_blob), args_blob_p))

  
  def Initialize(self, dictionary):
    args = messages_pb2.InitializeModelArgs()
    args.model_name = self.name()
    args.dictionary_name = dictionary.name();
    blob = args.SerializeToString()
    blob_p = ctypes.create_string_buffer(blob)
    HandleErrorCode(self.lib_,
                    self.lib_.ArtmInitializeModel(self.master_id_, len(blob), blob_p))

  def Overwrite(self, topic_model):
    copy_ = messages_pb2.TopicModel()
    copy_.CopyFrom(topic_model)
    copy_.name = self.name()
    blob = copy_.SerializeToString()
    blob_p = ctypes.create_string_buffer(blob)
    HandleErrorCode(self.lib_, 
                    self.lib_.ArtmOverwriteTopicModel(self.master_id_, len(blob), blob_p))

  def Enable(self):
    config_copy_ = messages_pb2.ModelConfig()
    config_copy_.CopyFrom(self.config_)
    config_copy_.enabled = True
    self.Reconfigure(config_copy_)

  def Disable(self):
    config_copy_ = messages_pb2.ModelConfig()
    config_copy_.CopyFrom(self.config_)
    config_copy_.enabled = False
    self.Reconfigure(config_copy_)

  def EnableScore(self, score):
    config_copy_ = messages_pb2.ModelConfig()
    config_copy_.CopyFrom(self.config_)
    config_copy_.score_name.append(score.name())
    self.Reconfigure(config_copy_)

  def EnableRegularizer(self, regularizer, tau):
    config_copy_ = messages_pb2.ModelConfig()
    config_copy_.CopyFrom(self.config_)
    config_copy_.regularizer_name.append(regularizer.name())
    config_copy_.regularizer_tau.append(tau)
    self.Reconfigure(config_copy_)

#################################################################################

class Regularizer:
  def __init__(self, master_component, config):
    self.lib_ = master_component.lib_
    self.master_id_ = master_component.id_
    self.config_ = config
    regularizer_config_blob = config.SerializeToString()
    regularizer_config_blob_p = ctypes.create_string_buffer(regularizer_config_blob)
    HandleErrorCode(self.lib_, self.lib_.ArtmCreateRegularizer(self.master_id_,
                    len(regularizer_config_blob), regularizer_config_blob_p))

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    __Dispose__(self)

  def __Dispose__(self):
    self.lib_.ArtmDisposeRegularizer(self.master_id_, self.config_.name)
    self.config_.name = ''
    self.master_id_ = -1

  def name(self):
    return self.config_.name

  def Reconfigure(self, type, config):
    general_config = messages_pb2.RegularizerConfig()
    general_config.name = self.name()
    general_config.type = type
    general_config.config = config.SerializeToString()

    regularizer_config_blob = general_config.SerializeToString()
    regularizer_config_blob_p = ctypes.create_string_buffer(regularizer_config_blob)
    HandleErrorCode(self.lib_, self.lib_.ArtmReconfigureRegularizer(self.master_id_,
                    len(regularizer_config_blob), regularizer_config_blob_p))
    self.config_.CopyFrom(general_config)

#################################################################################

class Dictionary:
  def __init__(self, master_component, config):
    self.lib_ = master_component.lib_
    self.master_id_ = master_component.id_
    self.config_ = config
    dictionary_config_blob = config.SerializeToString()
    dictionary_config_blob_p = ctypes.create_string_buffer(dictionary_config_blob)
    HandleErrorCode(self.lib_, self.lib_.ArtmCreateDictionary(self.master_id_,
                    len(dictionary_config_blob), dictionary_config_blob_p))

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    __Dispose__(self)

  def __Dispose__(self):
    self.lib_.ArtmDisposeDictionary(self.master_id_, self.config_.name)
    self.config_.name = ''
    self.master_id_ = -1

  def name(self):
    return self.config_.name

  def Reconfigure(self, config):
    dictionary_config_blob = config.SerializeToString()
    dictionary_config_blob_p = ctypes.create_string_buffer(dictionary_config_blob)
    HandleErrorCode(self.lib_, self.lib_.ArtmReconfigureDictionary(self.master_id_,
                    len(dictionary_config_blob), dictionary_config_blob_p))
    self.config_.CopyFrom(config)

#################################################################################

class Score:
  def __init__(self, master_component, score_name):
    self.master_id_  = master_component.id_
    self.lib_        = master_component.lib_
    self.score_name_ = score_name

  def name(self) :
    return self.score_name_

  def GetValue(self, model) :
    length = HandleErrorCode(self.lib_,
                             self.lib_.ArtmRequestScore(self.master_id_, model.name(), self.score_name_))
    blob = ctypes.create_string_buffer(length)
    HandleErrorCode(self.lib_, self.lib_.ArtmCopyRequestResult(length, blob))

    score_data = messages_pb2.ScoreData()
    score_data.ParseFromString(blob)

    if (score_data.type == ScoreData_Type_Perplexity):
      score = messages_pb2.PerplexityScore();
      score.ParseFromString(score_data.data);
      return score;
    elif (score_data.type == ScoreData_Type_SparsityTheta):
      score = messages_pb2.SparsityThetaScore();
      score.ParseFromString(score_data.data);
      return score;
    elif (score_data.type == ScoreData_Type_SparsityPhi):
      score = messages_pb2.SparsityPhiScore();
      score.ParseFromString(score_data.data);
      return score;
    elif (score_data.type == ScoreData_Type_ItemsProcessed):
      score = messages_pb2.ItemsProcessedScore()
      score.ParseFromString(score_data.data)
      return score
    elif (score_data.type == ScoreData_Type_TopTokens):
      score = messages_pb2.TopTokensScore()
      score.ParseFromString(score_data.data)
      return score
    elif (score_data.type == ScoreData_Type_ThetaSnippet):
      score = messages_pb2.ThetaSnippetScore()
      score.ParseFromString(score_data.data)
      return score
    elif (score_data.type == ScoreData_Type_TopicKernel):
      score = messages_pb2.TopicKernelScore()
      score.ParseFromString(score_data.data)
      return score

    # Unknown score type
    raise InvalidMessage(GetLastErrorMessage(self.lib_))

#################################################################################

class Visualizers:
  @staticmethod
  def PrintTopTokensScore(top_tokens_score):
    print '\nTop tokens per topic:',
    topic_index = -1
    for i in range(0, top_tokens_score.num_entries):
      if (top_tokens_score.topic_index[i] != topic_index):
        topic_index = top_tokens_score.topic_index[i]
        print "\nTopic#" + str(i+1) + ": ",
      print top_tokens_score.token[i] + "(%.2f) " % top_tokens_score.weight[i],

  @staticmethod
  def PrintThetaSnippetScore(theta_snippet_score):
    print '\nSnippet of theta matrix:'
    for i in range(0, len(theta_snippet_score.values)):
      print "Item#" + str(theta_snippet_score.item_id[i]) + ": ",
      for value in theta_snippet_score.values[i].value:
        print "%.3f\t" % value,
      print "\n",
