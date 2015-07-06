// Copyright 2014, Additive Regularization of Topic Models.
// This file defines public API methods of BigARTM library.
// All methods must be inside "extern "C"" scope. All complex data structures should be passed in
// as Google Protobuf Messages, defined in messages.proto.

#ifndef SRC_ARTM_C_INTERFACE_H_
#define SRC_ARTM_C_INTERFACE_H_

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
  DLL_PUBLIC int ArtmCreateMasterComponent(int length, const char* master_component_config);
  DLL_PUBLIC int ArtmReconfigureMasterComponent(int master_id, int length, const char* master_component_config);
  DLL_PUBLIC int ArtmDisposeMasterComponent(int master_id);

  DLL_PUBLIC int ArtmCreateModel(int master_id, int length, const char* model_config);
  DLL_PUBLIC int ArtmReconfigureModel(int master_id, int length, const char* model_config);
  DLL_PUBLIC int ArtmDisposeModel(int master_id, const char* model_name);

  DLL_PUBLIC int ArtmCreateRegularizer(int master_id, int length, const char* regularizer_config);
  DLL_PUBLIC int ArtmReconfigureRegularizer(int master_id, int length, const char* regularizer_config);
  DLL_PUBLIC int ArtmDisposeRegularizer(int master_id, const char* regularizer_name);

  DLL_PUBLIC int ArtmCreateDictionary(int master_id, int length, const char* dictionary_config);
  DLL_PUBLIC int ArtmReconfigureDictionary(int master_id, int length, const char* dictionary_config);
  DLL_PUBLIC int ArtmDisposeDictionary(int master_id, const char* dictionary_name);
  DLL_PUBLIC int ArtmImportDictionary(int master_id, int length, const char* import_dictionary_args);
  DLL_PUBLIC int ArtmParseCollection(int length, const char* collection_parser_config);

  DLL_PUBLIC int ArtmAddBatch(int master_id, int length, const char* add_batch_args);
  DLL_PUBLIC int ArtmInvokeIteration(int master_id, int length, const char* invoke_iteration_args);
  DLL_PUBLIC int ArtmWaitIdle(int master_id, int length, const char* wait_idle_args);
  DLL_PUBLIC int ArtmSynchronizeModel(int master_id, int length, const char* sync_model_args);

  DLL_PUBLIC int ArtmOverwriteTopicModel(int master_id, int length, const char* topic_model);
  DLL_PUBLIC int ArtmInitializeModel(int master_id, int length, const char* init_model_args);
  DLL_PUBLIC int ArtmExportModel(int master_id, int length, const char* export_model_args);
  DLL_PUBLIC int ArtmImportModel(int master_id, int length, const char* import_model_args);

  DLL_PUBLIC int ArtmRequestProcessBatches(int master_id, int length, const char* process_batches_args);
  DLL_PUBLIC int ArtmMergeModel(int master_id, int length, const char* merge_model_args);
  DLL_PUBLIC int ArtmRegularizeModel(int master_id, int length, const char* regularize_model_args);
  DLL_PUBLIC int ArtmNormalizeModel(int master_id, int length, const char* normalize_model_args);

  DLL_PUBLIC int ArtmRequestThetaMatrix(int master_id, int length, const char* get_theta_args);
  DLL_PUBLIC int ArtmRequestTopicModel(int master_id, int length, const char* get_model_args);
  DLL_PUBLIC int ArtmRequestRegularizerState(int master_id, const char* regularizer_name);
  DLL_PUBLIC int ArtmRequestScore(int master_id, int length, const char* get_score_args);
  DLL_PUBLIC int ArtmRequestParseCollection(int length, const char* collection_parser_config);
  DLL_PUBLIC int ArtmRequestLoadDictionary(const char* filename);
  DLL_PUBLIC int ArtmRequestLoadBatch(const char* filename);
  DLL_PUBLIC int ArtmCopyRequestResult(int length, char* address);
  DLL_PUBLIC int ArtmCopyRequestResultEx(int length, char* address, int args_length, const char* copy_result_args);

  DLL_PUBLIC int ArtmSaveBatch(const char* disk_path, int length, const char* batch);
  DLL_PUBLIC const char* ArtmGetLastErrorMessage();
}

#endif  // SRC_ARTM_C_INTERFACE_H_
