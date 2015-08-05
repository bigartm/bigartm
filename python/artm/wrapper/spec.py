"""
Specifications of C-API functions
"""

import ctypes

from . import messages_pb2


class CallSpec(object):
    def __init__(self, name, arguments, result=None, request=None):
        self.name = name
        self.arguments = arguments
        self.result_type = result
        self.request_type = request


ARTM_API = [
    CallSpec(
        'ArtmCreateMasterComponent',
        [('config', messages_pb2.MasterComponentConfig)],
        result=ctypes.c_int,
    ),
    CallSpec(
        'ArtmReconfigureMasterComponent',
        [('master_id', int), ('config', messages_pb2.MasterComponentConfig)],
    ),
    CallSpec(
        'ArtmDisposeMasterComponent',
        [('master_id', int)],
    ),
    CallSpec(
        'ArtmCreateModel',
        [('master_id', int), ('config', messages_pb2.ModelConfig)],
    ),
    CallSpec(
        'ArtmReconfigureModel',
        [('master_id', int), ('config', messages_pb2.ModelConfig)],
    ),
    CallSpec(
        'ArtmDisposeModel',
        [('master_id', int), ('name', str)],
    ),
    CallSpec(
        'ArtmCreateRegularizer',
        [('master_id', int), ('config', messages_pb2.RegularizerConfig)],
    ),
    CallSpec(
        'ArtmReconfigureRegularizer',
        [('master_id', int), ('config', messages_pb2.RegularizerConfig)],
    ),
    CallSpec(
        'ArtmDisposeRegularizer',
        [('master_id', int), ('name', str)],
    ),
    CallSpec(
        'ArtmCreateDictionary',
        [('master_id', int), ('config', messages_pb2.DictionaryConfig)],
    ),
    CallSpec(
        'ArtmReconfigureDictionary',
        [('master_id', int), ('config', messages_pb2.DictionaryConfig)],
    ),
    CallSpec(
        'ArtmDisposeDictionary',
        [('master_id', int), ('name', str)],
    ),
    CallSpec(
        'ArtmAddBatch',
        [('master_id', int), ('args', messages_pb2.AddBatchArgs)],
    ),
    CallSpec(
        'ArtmInvokeIteration',
        [('master_id', int), ('args', messages_pb2.InvokeIterationArgs)],
    ),
    CallSpec(
        'ArtmSynchronizeModel',
        [('master_id', int), ('args', messages_pb2.SynchronizeModelArgs)],
    ),
    CallSpec(
        'ArtmInitializeModel',
        [('master_id', int), ('args', messages_pb2.InitializeModelArgs)],
    ),
    CallSpec(
        'ArtmExportModel',
        [('master_id', int), ('args', messages_pb2.ExportModelArgs)],
    ),
    CallSpec(
        'ArtmImportModel',
        [('master_id', int), ('args', messages_pb2.ImportModelArgs)],
    ),
    CallSpec(
        'ArtmWaitIdle',
        [('master_id', int), ('args', messages_pb2.WaitIdleArgs)],
    ),
    CallSpec(
        'ArtmOverwriteTopicModel',
        [('master_id', int), ('model', messages_pb2.TopicModel)],
    ),
    CallSpec(
        'ArtmRequestThetaMatrix',
        [('master_id', int), ('args', messages_pb2.GetThetaMatrixArgs)],
        request=messages_pb2.ThetaMatrix,
    ),
    CallSpec(
        'ArtmRequestTopicModel',
        [('master_id', int), ('args', messages_pb2.GetTopicModelArgs)],
        request=messages_pb2.TopicModel,
    ),
    CallSpec(
        'ArtmRequestRegularizerState',
        [('master_id', int), ('name', str)],
        request=messages_pb2.RegularizerInternalState,
    ),
    CallSpec(
        'ArtmRequestScore',
        [('master_id', int), ('args', messages_pb2.GetScoreValueArgs)],
        request=messages_pb2.ScoreData,
    ),
    CallSpec(
        'ArtmRequestParseCollection',
        [('args', messages_pb2.CollectionParserConfig)],
        request=messages_pb2.DictionaryConfig,
    ),
    CallSpec(
        'ArtmRequestLoadDictionary',
        [('filename', str)],
        request=messages_pb2.DictionaryConfig,
    ),
    CallSpec(
        'ArtmRequestLoadBatch',
        [('filename', str)],
        request=messages_pb2.Batch,
    ),
    CallSpec(
        'ArtmSaveBatch',
        [('filename', str), ('batch', messages_pb2.Batch)],
    ),
    CallSpec(
        'ArtmSaveBatch',
        [('filename', str), ('batch', messages_pb2.Batch)],
    ),
]
