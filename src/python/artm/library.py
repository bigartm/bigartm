# Copyright 2014, Additive Regularization of Topic Models.

import sys
import os
import ctypes
import uuid
import glob

import messages_pb2


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
RegularizerConfig_Type_LabelRegularizationPhi = 4
RegularizerConfig_Type_SpecifiedSparsePhi = 5
RegularizerConfig_Type_ImproveCoherencePhi = 6
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
CollectionParserConfig_Format_MatrixMarket = 1
CollectionParserConfig_Format_VowpalWabbit = 2
MasterComponentConfig_ModusOperandi_Local = 0
MasterComponentConfig_ModusOperandi_Network = 1
GetTopicModelArgs_RequestType_Pwt = 0
GetTopicModelArgs_RequestType_Nwt = 1
InitializeModelArgs_SourceType_Dictionary = 0
InitializeModelArgs_SourceType_Batches = 1
SpecifiedSparsePhiConfig_Mode_SparseTopics = 0
SpecifiedSparsePhiConfig_Mode_SparseTokens = 1

#################################################################################

class InternalError(BaseException): pass
class ArgumentOutOfRangeException(BaseException): pass
class InvalidMasterIdException(BaseException): pass
class CorruptedMessageException(BaseException): pass
class InvalidOperationException(BaseException): pass
class DiskReadException(BaseException): pass
class DiskWriteException(BaseException): pass
class NetworkException(BaseException): pass


def GetLastErrorMessage(lib):
    lib.ArtmGetLastErrorMessage.restype = ctypes.c_char_p
    return lib.ArtmGetLastErrorMessage()


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
    def __init__(self, artm_shared_library=""):
        if not artm_shared_library:
            if sys.platform.count('linux') == 1:
                artm_shared_library = 'libartm.so'
            elif sys.platform.count('darwin') == 1:
                artm_shared_library = 'libartm.dylib'
            else:
                artm_shared_library = 'artm.dll'

        try:
            self.lib_ = ctypes.CDLL(artm_shared_library)
            return
        except OSError as e:
            print >> sys.stderr, str(e) + ", fall back to ARTM_SHARED_LIBRARY environment variable"

        if "ARTM_SHARED_LIBRARY" in os.environ:
            try:
                self.lib_ = ctypes.CDLL(os.environ['ARTM_SHARED_LIBRARY'])
                return
            except OSError as e:
                print >> sys.stderr, str(e)

        print "Failed to load artm shared library, try to set ARTM_SHARED_LIBRARY environment variable"
        sys.exit(1)


    def CreateMasterComponent(self, config=None):
        if config is None:
            config = messages_pb2.MasterComponentConfig()
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
            collection_parser_config = messages_pb2.CollectionParserConfig()
            collection_parser_config.format = CollectionParserConfig_Format_BagOfWordsUci

            collection_parser_config.docword_file_path = docword_file_path
            collection_parser_config.vocab_file_path = vocab_file_path
            collection_parser_config.target_folder = target_folder
            collection_parser_config.dictionary_file_name = 'dictionary'
            unique_tokens = self.ParseCollection(collection_parser_config)
            print " OK."
            return unique_tokens
        else:
            print "Found " + str(batches_found) + " batches, using them."
            return self.LoadDictionary(target_folder + '/dictionary')

#################################################################################


class MasterComponent:
    def __init__(self, config=None, lib=None, disk_path=None):
        if lib is None:
            lib = Library().lib_

        if config is None:
            config = messages_pb2.MasterComponentConfig()
        if disk_path is not None:
            config.disk_path = disk_path

        self.lib_ = lib
        master_config_blob = config.SerializeToString()
        master_config_blob_p = ctypes.create_string_buffer(master_config_blob)

        if isinstance(config, messages_pb2.MasterComponentConfig):
            self.config_ = config
            self.id_ = HandleErrorCode(self.lib_, self.lib_.ArtmCreateMasterComponent(
                len(master_config_blob), master_config_blob_p))
            return

        raise ArgumentOutOfRangeException("config must be MasterComponentConfig")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.Dispose()

    def Dispose(self):
        self.lib_.ArtmDisposeMasterComponent(self.id_)
        self.id_ = -1

    def config(self):
        return self.config_

    def CreateModel(self, config=None, topics_count=None, inner_iterations_count=None, topic_names=None,
                    class_ids=None, class_weights=None):
        if config is None:
            config = messages_pb2.ModelConfig()
        if topics_count is not None:
            config.topics_count = topics_count
        if inner_iterations_count is not None:
            config.inner_iterations_count = inner_iterations_count
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
        if class_ids is not None:
            config.ClearField('class_id')
            for class_id in class_ids:
                config.class_id.append(class_id)
        if class_weights is not None:
            config.ClearField('class_weight')
            for class_weight in class_weights:
                config.class_weight.append(class_weight)
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

    def CreateSmoothSparseThetaRegularizer(self, name=None, config=None, topic_names=None):
        if name is None:
            name = "SmoothSparseThetaRegularizer:" + uuid.uuid1().urn
        if config is None:
            config = messages_pb2.SmoothSparseThetaConfig()
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
        return self.CreateRegularizer(name, RegularizerConfig_Type_SmoothSparseTheta, config)

    def CreateSmoothSparsePhiRegularizer(self, name=None, config=None, topic_names=None, class_ids=None):
        if name is None:
            name = "SmoothSparsePhiRegularizer:" + uuid.uuid1().urn
        if config is None:
            config = messages_pb2.SmoothSparsePhiConfig()
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
        if class_ids is not None:
            config.ClearField('class_id')
            for class_id in class_ids:
                config.class_id.append(class_id)
        return self.CreateRegularizer(name, RegularizerConfig_Type_SmoothSparsePhi, config)

    def CreateDecorrelatorPhiRegularizer(self, name=None, config=None, topic_names=None, class_ids=None):
        if name is None:
            name = "DecorrelatorPhiRegularizer:" + uuid.uuid1().urn
        if config is None:
            config = messages_pb2.DecorrelatorPhiConfig()
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
        if class_ids is not None:
            config.ClearField('class_id')
            for class_id in class_ids:
                config.class_id.append(class_id)
        return self.CreateRegularizer(name, RegularizerConfig_Type_DecorrelatorPhi, config)

    def CreateLabelRegularizationPhiRegularizer(self, name=None, config=None, topic_names=None, class_ids=None):
        if name is None:
            name = "LabelRegularizationPhiRegularizer:" + uuid.uuid1().urn
        if config is None:
            config = messages_pb2.LabelRegularizationPhiConfig()
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
        if class_ids is not None:
            config.ClearField('class_id')
            for class_id in class_ids:
                config.class_id.append(class_id)
        return self.CreateRegularizer(name, RegularizerConfig_Type_LabelRegularizationPhi, config)

    def CreateSpecifiedSparsePhiRegularizer(self, name=None, config=None, topic_names=None, class_id=None, mode=None):
        if name is None:
            name = "SpecifiedSparsePhiRegularizer:" + uuid.uuid1().urn
        if config is None:
            config = messages_pb2.SpecifiedSparsePhiConfig()
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
        if class_id is not None:
            config.class_id = class_id
        if mode is not None:
            config.mode = mode
        return self.CreateRegularizer(name, RegularizerConfig_Type_SpecifiedSparsePhi, config)

    def CreateImproveCoherencePhiRegularizer(self, name=None, config=None, topic_names=None,
                                             class_ids=None, dictionary_name=None):
        if name is None:
            name = "ImproveCoherencePhiRegularizer:" + uuid.uuid1().urn
        if config is None:
            config = messages_pb2.ImproveCoherencePhiConfig()
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
        if class_ids is not None:
            config.ClearField('class_id')
            for class_id in class_ids:
                config.class_id.append(class_id)
        if dictionary_name is not None:
            config.dictionary_name = dictionary_name
        return self.CreateRegularizer(name, RegularizerConfig_Type_ImproveCoherencePhi, config)

    def CreateScore(self, name, type, config):
        master_config = messages_pb2.MasterComponentConfig()
        master_config.CopyFrom(self.config_)
        score_config = master_config.score_config.add()
        score_config.name = name
        score_config.type = type
        score_config.config = config.SerializeToString()
        self.Reconfigure(master_config)
        return Score(self, name)

    def CreatePerplexityScore(self, name=None, config=None, stream_name=None, class_ids=None):
        if config is None:
            config = messages_pb2.PerplexityScoreConfig()
        if name is None:
            name = "PerplexityScore:" + uuid.uuid1().urn
        if stream_name is not None:
            config.stream_name = stream_name
        if class_ids is not None:
            config.ClearField('class_id')
            for class_id in class_ids:
                config.class_id.append(class_id)
        return self.CreateScore(name, ScoreConfig_Type_Perplexity, config)

    def CreateSparsityThetaScore(self, name=None, config=None, topic_names=None):
        if config is None:
            config = messages_pb2.SparsityThetaScoreConfig()
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
        if name is None:
            name = "SparsityThetaScore:" + uuid.uuid1().urn
        return self.CreateScore(name, ScoreConfig_Type_SparsityTheta, config)

    def CreateSparsityPhiScore(self, name=None, config=None, topic_names=None, class_id=None):
        if config is None:
            config = messages_pb2.SparsityPhiScoreConfig()
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
        if class_id is not None:
            config.class_id = class_id
        if name is None:
            name = "SparsityPhiScore:" + uuid.uuid1().urn

        return self.CreateScore(name, ScoreConfig_Type_SparsityPhi, config)

    def CreateItemsProcessedScore(self, name=None, config=None):
        if config is None:
            config = messages_pb2.ItemsProcessedScoreConfig()
        if name is None:
            name = "ItemsProcessedScore:" + uuid.uuid1().urn
        return self.CreateScore(name, ScoreConfig_Type_ItemsProcessed, config)

    def CreateTopTokensScore(self, name=None, config=None, num_tokens=None,
                             class_id=None, topic_names=None):
        if config is None:
            config = messages_pb2.TopTokensScoreConfig()
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
        if num_tokens is not None:
            config.num_tokens = num_tokens
        if class_id is not None:
            config.class_id = class_id
        if name is None:
            name = "TopTokensScore:" + uuid.uuid1().urn
        return self.CreateScore(name, ScoreConfig_Type_TopTokens, config)

    def CreateThetaSnippetScore(self, name=None, config=None, topic_names=None):
        if config is None:
            config = messages_pb2.ThetaSnippetScoreConfig()
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
        if name is None:
            name = "ThetaSnippetScore:" + uuid.uuid1().urn
        return self.CreateScore(name, ScoreConfig_Type_ThetaSnippet, config)

    def CreateTopicKernelScore(self, name=None, config=None, topic_names=None, class_id=None):
        if config is None:
            config = messages_pb2.TopicKernelScoreConfig()
        if topic_names is not None:
            config.ClearField('topic_name')
            for topic_name in topic_names:
                config.topic_name.append(topic_name)
        if class_id is not None:
            config.class_id = class_id
        if name is None:
            name = "TopicKernelScore:" + uuid.uuid1().urn
        return self.CreateScore(name, ScoreConfig_Type_TopicKernel, config)

    def RemoveScore(self, name):
        raise NotImplementedError

    def CreateDictionary(self, config):
        return Dictionary(self, config)

    def RemoveDictionary(self, dictionary):
        dictionary.__Dispose__()

    def Reconfigure(self, config=None):
        if config is None:
            config = self.config_
        config_blob = config.SerializeToString()
        config_blob_p = ctypes.create_string_buffer(config_blob)
        HandleErrorCode(self.lib_, self.lib_.ArtmReconfigureMasterComponent(self.id_, len(config_blob), config_blob_p))
        self.config_.CopyFrom(config)

    def AddBatch(self, batch=None, batch_filename=None, timeout=-1, reset_scores=False, args=None):
        if args is None:
            args = messages_pb2.AddBatchArgs()
        if batch is not None:
            args.batch.CopyFrom(batch)
        args.timeout_milliseconds = timeout
        args.reset_scores = reset_scores
        if batch_filename is not None:
            args.batch_file_name = batch_filename
        args_blob = args.SerializeToString()
        args_blob_p = ctypes.create_string_buffer(args_blob)

        result = self.lib_.ArtmAddBatch(self.id_, len(args_blob), args_blob_p)
        result = HandleErrorCode(self.lib_, result)
        return False if (result == ARTM_STILL_WORKING) else True

    def InvokeIteration(self, iterations_count=1, disk_path=None, args=None):
        if args is None:
            args = messages_pb2.InvokeIterationArgs()
        args.iterations_count = iterations_count
        if disk_path is not None:
            args.disk_path = disk_path
        args_blob = args.SerializeToString()
        args_blob_p = ctypes.create_string_buffer(args_blob)
        HandleErrorCode(self.lib_, self.lib_.ArtmInvokeIteration(self.id_, len(args_blob), args_blob_p))

    def WaitIdle(self, timeout=None, args=None):
        if args is None:
            args = messages_pb2.WaitIdleArgs()
        if timeout is not None:
            args.timeout_milliseconds = timeout
        args_blob = args.SerializeToString()
        args_blob_p = ctypes.create_string_buffer(args_blob)

        result = self.lib_.ArtmWaitIdle(self.id_, len(args_blob), args_blob_p)
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
            if self.config_.stream[stream_index].name != stream_name:
                s = new_config_.stream.add()
                s.CopyFrom(self.config_.stream[stream_index])
        self.Reconfigure(new_config_)

    def GetTopicModel(self, model=None, args=None, class_ids=None, topic_names=None, use_sparse_format=None,
                      request_type=None):
        if args is None:
            args = messages_pb2.GetTopicModelArgs()
        if model is not None:
            args.model_name = model.name()
        if class_ids is not None:
            args.ClearField('class_id')
            for class_id in class_ids:
                args.class_id.append(class_id)
        if topic_names is not None:
            args.ClearField('topic_name')
            for topic_name in topic_names:
                args.topic_name.append(topic_name)
        if use_sparse_format is not None:
            args.use_sparse_format=use_sparse_format
        if request_type is not None:
            args.request_type = request_type

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

    def GetThetaMatrix(self, model=None, batch=None, clean_cache=None, args=None, topic_names=None):
        if args is None:
            args = messages_pb2.GetThetaMatrixArgs()
        if model is not None:
            args.model_name = model.name()
        if batch is not None:
            args.batch.CopyFrom(batch)
        if clean_cache is not None:
            args.clean_cache = clean_cache
        if topic_names is not None:
            args.ClearField('topic_name')
            for topic_name in topic_names:
                args.topic_name.append(topic_name)
        args_blob = args.SerializeToString()
        length = HandleErrorCode(self.lib_, self.lib_.ArtmRequestThetaMatrix(self.id_, len(args_blob), args_blob))
        blob = ctypes.create_string_buffer(length)
        HandleErrorCode(self.lib_, self.lib_.ArtmCopyRequestResult(length, blob))

        theta_matrix = messages_pb2.ThetaMatrix()
        theta_matrix.ParseFromString(blob)
        return theta_matrix

#################################################################################

class NodeController:
    def __init__(self, endpoint, lib=None):
        config = messages_pb2.NodeControllerConfig()
        config.create_endpoint = endpoint

        if lib is None:
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
        self.Dispose()

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

    def Reconfigure(self, config=None):
        if config is None:
            config = self.config_

        model_config_blob = config.SerializeToString()
        model_config_blob_p = ctypes.create_string_buffer(model_config_blob)
        HandleErrorCode(self.lib_, self.lib_.ArtmReconfigureModel(self.master_id_,
                                                                  len(model_config_blob), model_config_blob_p))
        self.config_.CopyFrom(config)

    def Synchronize(self, decay_weight=0.0, apply_weight=1.0, invoke_regularizers=True, args=None):
        if args is None:
            args = messages_pb2.SynchronizeModelArgs()
        args.model_name = self.name()
        args.decay_weight = decay_weight
        args.apply_weight = apply_weight
        args.invoke_regularizers = invoke_regularizers
        args_blob = args.SerializeToString()
        args_blob_p = ctypes.create_string_buffer(args_blob)
        HandleErrorCode(self.lib_, self.lib_.ArtmSynchronizeModel(
            self.master_id_, len(args_blob), args_blob_p))

    def Initialize(self, dictionary=None, args=None):
        if args is None:
            args = messages_pb2.InitializeModelArgs()
        args.model_name = self.name()
        if dictionary is not None:
            args.dictionary_name = dictionary.name()
            args.source_type = InitializeModelArgs_SourceType_Dictionary
        blob = args.SerializeToString()
        blob_p = ctypes.create_string_buffer(blob)
        HandleErrorCode(self.lib_,
                        self.lib_.ArtmInitializeModel(self.master_id_, len(blob), blob_p))

    def Overwrite(self, topic_model, commit=True):
        copy_ = messages_pb2.TopicModel()
        copy_.CopyFrom(topic_model)
        copy_.name = self.name()
        blob = copy_.SerializeToString()
        blob_p = ctypes.create_string_buffer(blob)
        HandleErrorCode(self.lib_,
                        self.lib_.ArtmOverwriteTopicModel(self.master_id_, len(blob), blob_p))

        if commit:
            self.master_component.WaitIdle()
            self.Synchronize(decay_weight=0.0, apply_weight=1.0, invoke_regularizers=False)

    def Export(self, filename):
        args = messages_pb2.ExportModelArgs()
        args.model_name = self.name()
        args.file_name = filename
        blob = args.SerializeToString()
        blob_p = ctypes.create_string_buffer(blob)
        HandleErrorCode(self.lib_,
                        self.lib_.ArtmExportModel(self.master_id_, len(blob), blob_p))

    def Import(self, filename):
        args = messages_pb2.ImportModelArgs()
        args.model_name = self.name()
        args.file_name = filename
        blob = args.SerializeToString()
        blob_p = ctypes.create_string_buffer(blob)
        HandleErrorCode(self.lib_,
                        self.lib_.ArtmImportModel(self.master_id_, len(blob), blob_p))

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

    def EnableScore(self, score):  # obsolete in BigARTM v0.6.3
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
                                                                   len(regularizer_config_blob),
                                                                   regularizer_config_blob_p))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.Dispose()

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
                                                                        len(regularizer_config_blob),
                                                                        regularizer_config_blob_p))
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
                                                                  len(dictionary_config_blob),
                                                                  dictionary_config_blob_p))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.Dispose()

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
                                                                       len(dictionary_config_blob),
                                                                       dictionary_config_blob_p))
        self.config_.CopyFrom(config)

#################################################################################


class Score:
    def __init__(self, master_component, score_name):
        self.master_id_ = master_component.id_
        self.lib_ = master_component.lib_
        self.score_name_ = score_name

    def name(self):
        return self.score_name_

    def GetValue(self, model=None, batch=None):
        args = messages_pb2.GetScoreValueArgs()
        args.score_name = self.score_name_
        if model is not None:
            args.model_name = model.name()
        if batch is not None:
            args.batch.CopyFrom(batch)
        args_blob = args.SerializeToString()
        length = HandleErrorCode(self.lib_,
                                 self.lib_.ArtmRequestScore(self.master_id_, len(args_blob), args_blob))
        blob = ctypes.create_string_buffer(length)
        HandleErrorCode(self.lib_, self.lib_.ArtmCopyRequestResult(length, blob))

        score_data = messages_pb2.ScoreData()
        score_data.ParseFromString(blob)

        if score_data.type == ScoreData_Type_Perplexity:
            score = messages_pb2.PerplexityScore()
            score.ParseFromString(score_data.data)
            return score
        elif score_data.type == ScoreData_Type_SparsityTheta:
            score = messages_pb2.SparsityThetaScore()
            score.ParseFromString(score_data.data)
            return score
        elif score_data.type == ScoreData_Type_SparsityPhi:
            score = messages_pb2.SparsityPhiScore()
            score.ParseFromString(score_data.data)
            return score
        elif score_data.type == ScoreData_Type_ItemsProcessed:
            score = messages_pb2.ItemsProcessedScore()
            score.ParseFromString(score_data.data)
            return score
        elif score_data.type == ScoreData_Type_TopTokens:
            score = messages_pb2.TopTokensScore()
            score.ParseFromString(score_data.data)
            return score
        elif score_data.type == ScoreData_Type_ThetaSnippet:
            score = messages_pb2.ThetaSnippetScore()
            score.ParseFromString(score_data.data)
            return score
        elif score_data.type == ScoreData_Type_TopicKernel:
            score = messages_pb2.TopicKernelScore()
            score.ParseFromString(score_data.data)
            return score

        # Unknown score type
        raise ArgumentOutOfRangeException("ScoreData.type")

#################################################################################


class Visualizers:
    @staticmethod
    def PrintTopTokensScore(top_tokens_score):
        print 'Top tokens per topic:',
        topic_index = -1
        for i in range(0, top_tokens_score.num_entries):
            if top_tokens_score.topic_index[i] != topic_index:
                topic_index = top_tokens_score.topic_index[i]
                print "\nTopic#" + str(topic_index + 1) + ": ",
            print top_tokens_score.token[i] + "(%.3f) " % top_tokens_score.weight[i],
        print '\n',

    @staticmethod
    def PrintThetaSnippetScore(theta_snippet_score):
        print '\nSnippet of theta matrix:'
        for i in range(0, len(theta_snippet_score.values)):
            print "Item#" + str(theta_snippet_score.item_id[i]) + ": ",
            for value in theta_snippet_score.values[i].value:
                print "%.3f\t" % value,
            print "\n",
