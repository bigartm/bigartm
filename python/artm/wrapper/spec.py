"""
Specifications of C-API functions
"""

# TODO: generate this file automatically

import ctypes

import numpy

from . import messages_pb2 as messages


class CallSpec(object):
    def __init__(self, name, arguments, result=None, request=None):
        self.name = name
        self.arguments = arguments
        self.result_type = result
        self.request_type = request


ARTM_API = [

    CallSpec(
        'ArtmDuplicateMasterComponent',
        [('master_id', int), ('config', messages.DuplicateMasterComponentArgs)],
        result=ctypes.c_int,
    ),
    CallSpec(
        'ArtmDisposeMasterComponent',
        [('master_id', int)],
    ),
    CallSpec(
        'ArtmDisposeModel',
        [('master_id', int), ('name', str)],
    ),
    CallSpec(
        'ArtmCreateRegularizer',
        [('master_id', int), ('config', messages.RegularizerConfig)],
    ),
    CallSpec(
        'ArtmReconfigureRegularizer',
        [('master_id', int), ('config', messages.RegularizerConfig)],
    ),
    CallSpec(
        'ArtmDisposeRegularizer',
        [('master_id', int), ('name', str)],
    ),
    CallSpec(
        'ArtmCreateDictionary',
        [('master_id', int), ('config', messages.DictionaryData)],
    ),
    CallSpec(
        'ArtmDisposeDictionary',
        [('master_id', int), ('name', str)],
    ),
    CallSpec(
        'ArtmGatherDictionary',
        [('master_id', int), ('config', messages.GatherDictionaryArgs)],
    ),
    CallSpec(
        'ArtmFilterDictionary',
        [('master_id', int), ('config', messages.FilterDictionaryArgs)],
    ),
    CallSpec(
        'ArtmImportDictionary',
        [('master_id', int), ('args', messages.ImportDictionaryArgs)],
    ),
    CallSpec(
        'ArtmExportDictionary',
        [('master_id', int), ('args', messages.ExportDictionaryArgs)],
    ),
    CallSpec(
        'ArtmParseCollection',
        [('config', messages.CollectionParserConfig)],
    ),
    CallSpec(
        'ArtmImportBatches',
        [('master_id', int), ('args', messages.ImportBatchesArgs)],
    ),
    CallSpec(
        'ArtmClearThetaCache',
        [('master_id', int), ('args', messages.ClearThetaCacheArgs)],
    ),
    CallSpec(
        'ArtmClearScoreCache',
        [('master_id', int), ('args', messages.ClearScoreCacheArgs)],
    ),
    CallSpec(
        'ArtmDisposeBatch',
        [('master_id', int), ('name', str)],
    ),

    ## deprecated
    #CallSpec(
    #    'ArtmAddBatch',
    #    [('master_id', int), ('args', messages.AddBatchArgs)],
    #),
    ## deprecated
    #CallSpec(
    #    'ArtmInvokeIteration',
    #    [('master_id', int), ('args', messages.InvokeIterationArgs)],
    #),
    ## deprecated
    #CallSpec(
    #    'ArtmWaitIdle',
    #    [('master_id', int), ('args', messages.WaitIdleArgs)],
    #),
    ## deprecated
    #CallSpec(
    #    'ArtmSynchronizeModel',
    #    [('master_id', int), ('args', messages.SynchronizeModelArgs)],
    #),

    ## deprecated
    #CallSpec(
    #    'ArtmOverwriteTopicModel',
    #    [('master_id', int), ('model', messages.TopicModel)],
    #),
    CallSpec(
        'ArtmInitializeModel',
        [('master_id', int), ('args', messages.InitializeModelArgs)],
    ),
    CallSpec(
        'ArtmExportModel',
        [('master_id', int), ('args', messages.ExportModelArgs)],
    ),
    CallSpec(
        'ArtmImportModel',
        [('master_id', int), ('args', messages.ImportModelArgs)],
    ),
    CallSpec(
        'ArtmAttachModel',
        [('master_id', int), ('args', messages.AttachModelArgs), ('matrix', numpy.ndarray)],
    ),
    CallSpec(
        'ArtmRequestProcessBatches',
        [('master_id', int), ('args', messages.ProcessBatchesArgs)],
        request=messages.ProcessBatchesResult,
    ),
    CallSpec(
        'ArtmRequestProcessBatchesExternal',
        [('master_id', int), ('args', messages.ProcessBatchesArgs)],
        request=messages.ProcessBatchesResult,
    ),
    CallSpec(
        'ArtmMergeModel',
        [('master_id', int), ('args', messages.MergeModelArgs)],
    ),
    CallSpec(
        'ArtmRegularizeModel',
        [('master_id', int), ('args', messages.RegularizeModelArgs)],
    ),
    CallSpec(
        'ArtmNormalizeModel',
        [('master_id', int), ('args', messages.NormalizeModelArgs)],
    ),
    CallSpec(
        'ArtmRequestThetaMatrix',
        [('master_id', int), ('args', messages.GetThetaMatrixArgs)],
        request=messages.ThetaMatrix,
    ),
    CallSpec(
        'ArtmRequestThetaMatrixExternal',
        [('master_id', int), ('args', messages.GetThetaMatrixArgs)],
        request=messages.ThetaMatrix,
    ),
    CallSpec(
        'ArtmRequestTopicModel',
        [('master_id', int), ('args', messages.GetTopicModelArgs)],
        request=messages.TopicModel,
    ),
    CallSpec(
        'ArtmRequestTopicModelExternal',
        [('master_id', int), ('args', messages.GetTopicModelArgs)],
        request=messages.TopicModel,
    ),
    CallSpec(
        'ArtmRequestScore',
        [('master_id', int), ('args', messages.GetScoreValueArgs)],
        request=messages.ScoreData,
    ),
    CallSpec(
        'ArtmRequestLoadBatch',
        [('filename', str)],
        request=messages.Batch,
    ),
    CallSpec(
        'ArtmRequestDictionary',
        [('master_id', int), ('args', messages.GetDictionaryArgs)],
        request=messages.DictionaryData,
    ),
    CallSpec(
        'ArtmSaveBatch',
        [('filename', str), ('batch', messages.Batch)],
    ),
    CallSpec(
        'ArtmUpgradeBatch_v07',
        [('source_path', str), ('target_path', str)],
    ),
    CallSpec(
        'ArtmCopyRequestResultEx',
        [('array', numpy.ndarray), ('args', messages.CopyRequestResultArgs)],
    ),
    CallSpec(
        'ArtmCreateMasterModel',
        [('config', messages.MasterModelConfig)],
        result=ctypes.c_int,
    ),
    CallSpec(
        'ArtmReconfigureMasterModel',
        [('master_id', int), ('config', messages.MasterModelConfig)],
    ),
    CallSpec(
        'ArtmFitOfflineMasterModel',
        [('master_id', int), ('config', messages.FitOfflineMasterModelArgs)],
    ),
    CallSpec(
        'ArtmFitOnlineMasterModel',
        [('master_id', int), ('config', messages.FitOnlineMasterModelArgs)],
    ),
    CallSpec(
        'ArtmRequestTransformMasterModel',
        [('master_id', int), ('config', messages.TransformMasterModelArgs)],
        request=messages.ThetaMatrix,
    ),
    CallSpec(
        'ArtmRequestTransformMasterModelExternal',
        [('master_id', int), ('config', messages.TransformMasterModelArgs)],
        request=messages.ThetaMatrix,
    ),
]
