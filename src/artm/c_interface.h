// Copyright 2017, Additive Regularization of Topic Models.

// This file defines public API methods of BigARTM library.
// All methods must be inside "extern "C"" scope. All complex data structures should be passed in
// as Google Protobuf Messages, defined in messages.proto.

#pragma once

#include <stdint.h>

#if defined(WIN32)
  #ifdef artm_EXPORTS
    #define DLL_PUBLIC __declspec(dllexport)
  #else
    #define DLL_PUBLIC __declspec(dllimport)
  #endif
#else
  #define DLL_PUBLIC
#endif

extern "C" {
  DLL_PUBLIC int64_t ArtmDuplicateMasterComponent(int master_id, int64_t length, const char* duplicate_master_args);
  DLL_PUBLIC int64_t ArtmCreateMasterModel(int64_t length, const char* master_model_config);
  DLL_PUBLIC int64_t ArtmReconfigureMasterModel(int master_id, int64_t length, const char* master_model_config);
  DLL_PUBLIC int64_t ArtmReconfigureTopicName(int master_id, int64_t length, const char* master_model_config);
  DLL_PUBLIC int64_t ArtmDisposeMasterComponent(int master_id);

  DLL_PUBLIC int64_t ArtmDisposeModel(int master_id, const char* model_name);

  DLL_PUBLIC int64_t ArtmClearThetaCache(int master_id, int64_t length, const char* clear_theta_cache_args);
  DLL_PUBLIC int64_t ArtmClearScoreCache(int master_id, int64_t length, const char* clear_score_cache_args);
  DLL_PUBLIC int64_t ArtmClearScoreArrayCache(int master_id, int64_t length, const char* clear_score_array_cache_args);

  DLL_PUBLIC int64_t ArtmCreateRegularizer(int master_id, int64_t length, const char* regularizer_config);
  DLL_PUBLIC int64_t ArtmReconfigureRegularizer(int master_id, int64_t length, const char* regularizer_config);
  DLL_PUBLIC int64_t ArtmDisposeRegularizer(int master_id, const char* regularizer_name);

  DLL_PUBLIC int64_t ArtmGatherDictionary(int master_id, int64_t length, const char* gather_dictionary_args);
  DLL_PUBLIC int64_t ArtmFilterDictionary(int master_id, int64_t length, const char* filter_dictionary_args);
  DLL_PUBLIC int64_t ArtmCreateDictionary(int master_id, int64_t length, const char* dictionary_data);
  DLL_PUBLIC int64_t ArtmCreateDictionaryNamed(int master_id, int64_t length,
                                               const char* dictionary_data, const char* name);
  DLL_PUBLIC int64_t ArtmRequestDictionary(int master_id, int64_t length, const char* request_dictionary_args);
  DLL_PUBLIC int64_t ArtmDisposeDictionary(int master_id, const char* dictionary_name);

  DLL_PUBLIC int64_t ArtmImportDictionary(int master_id, int64_t length, const char* import_dictionary_args);
  DLL_PUBLIC int64_t ArtmExportDictionary(int master_id, int64_t length, const char* export_dictionary_args);
  DLL_PUBLIC int64_t ArtmParseCollection(int64_t length, const char* collection_parser_config);

  DLL_PUBLIC int64_t ArtmImportBatches(int master_id, int64_t length, const char* import_batches_args);
  DLL_PUBLIC int64_t ArtmDisposeBatch(int master_id, const char* batch_name);

  DLL_PUBLIC int64_t ArtmOverwriteTopicModel(int master_id, int64_t length, const char* topic_model);
  DLL_PUBLIC int64_t ArtmOverwriteTopicModelNamed(int master_id, int64_t length,
                                                  const char* topic_model, const char* name);
  DLL_PUBLIC int64_t ArtmInitializeModel(int master_id, int64_t length, const char* init_model_args);
  DLL_PUBLIC int64_t ArtmExportModel(int master_id, int64_t length, const char* export_model_args);
  DLL_PUBLIC int64_t ArtmImportModel(int master_id, int64_t length, const char* import_model_args);
  DLL_PUBLIC int64_t ArtmAttachModel(int master_id, int64_t length, const char* attach_model_args,
                                     int64_t address_length, char* address);

  DLL_PUBLIC int64_t ArtmRequestProcessBatches(int master_id, int64_t length, const char* process_batches_args);
  DLL_PUBLIC int64_t ArtmRequestProcessBatchesExternal(int master_id, int64_t length, const char* process_batches_args);
  DLL_PUBLIC int64_t ArtmAsyncProcessBatches(int master_id, int64_t length, const char* process_batches_args);
  DLL_PUBLIC int64_t ArtmMergeModel(int master_id, int64_t length, const char* merge_model_args);
  DLL_PUBLIC int64_t ArtmRegularizeModel(int master_id, int64_t length, const char* regularize_model_args);
  DLL_PUBLIC int64_t ArtmNormalizeModel(int master_id, int64_t length, const char* normalize_model_args);

  DLL_PUBLIC int64_t ArtmFitOfflineMasterModel(int master_id, int64_t length,
                                               const char* fit_offline_master_model_args);
  DLL_PUBLIC int64_t ArtmFitOnlineMasterModel(int master_id, int64_t length, const char* fit_online_master_model_args);
  DLL_PUBLIC int64_t ArtmRequestTransformMasterModel(int master_id, int64_t length,
                                                     const char* transform_master_model_args);
  DLL_PUBLIC int64_t ArtmRequestTransformMasterModelExternal(int master_id, int64_t length,
                                                             const char* transform_master_model_args);

  DLL_PUBLIC int64_t ArtmRequestMasterModelConfig(int master_id);

  DLL_PUBLIC int64_t ArtmRequestThetaMatrix(int master_id, int64_t length, const char* get_theta_args);
  DLL_PUBLIC int64_t ArtmRequestThetaMatrixExternal(int master_id, int64_t length, const char* get_theta_args);
  DLL_PUBLIC int64_t ArtmRequestTopicModel(int master_id, int64_t length, const char* get_model_args);
  DLL_PUBLIC int64_t ArtmRequestTopicModelExternal(int master_id, int64_t length, const char* get_model_args);

  DLL_PUBLIC int64_t ArtmRequestScore(int master_id, int64_t length, const char* get_score_args);
  DLL_PUBLIC int64_t ArtmRequestScoreArray(int master_id, int64_t length, const char* get_score_args);

  DLL_PUBLIC int64_t ArtmExportScoreTracker(int master_id, int64_t length, const char* export_score_tracker_args);
  DLL_PUBLIC int64_t ArtmImportScoreTracker(int master_id, int64_t length, const char* import_score_tracker_args);

  DLL_PUBLIC int64_t ArtmRequestMasterComponentInfo(int master_id, int64_t length, const char* get_master_info_args);
  DLL_PUBLIC int64_t ArtmRequestLoadBatch(const char* filename);
  DLL_PUBLIC int64_t ArtmCopyRequestedMessage(int64_t length, char* address);
  DLL_PUBLIC int64_t ArtmCopyRequestedObject(int64_t length, char* address);

  DLL_PUBLIC int64_t ArtmAwaitOperation(int operation_id, int64_t length, const char* await_operation_args);

  DLL_PUBLIC int64_t ArtmSaveBatch(const char* disk_path, int64_t length, const char* batch);
  DLL_PUBLIC const char* ArtmGetLastErrorMessage();
  DLL_PUBLIC const char* ArtmGetVersion();
  DLL_PUBLIC int64_t ArtmConfigureLogging(int64_t length, const char* configure_logging_args);

  DLL_PUBLIC int64_t ArtmSetProtobufMessageFormatToJson();
  DLL_PUBLIC int64_t ArtmSetProtobufMessageFormatToBinary();
  DLL_PUBLIC int64_t ArtmProtobufMessageFormatIsJson();
}
