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
  DLL_PUBLIC int ArtmCreateMasterProxy(int length, const char* master_proxy_config);
  DLL_PUBLIC int ArtmReconfigureMasterComponent(int master_id, int length, const char* master_component_config);
  DLL_PUBLIC int ArtmDisposeMasterComponent(int master_id);

  DLL_PUBLIC int ArtmCreateNodeController(int length, const char* node_controller_config);
  DLL_PUBLIC int ArtmDisposeNodeController(int node_controller_id);

  DLL_PUBLIC int ArtmCreateModel(int master_id, int length, const char* model_config);
  DLL_PUBLIC int ArtmReconfigureModel(int master_id, int length, const char* model_config);
  DLL_PUBLIC int ArtmDisposeModel(int master_id, const char* model_name);

  DLL_PUBLIC int ArtmCreateRegularizer(int master_id, int length, const char* regularizer_config);
  DLL_PUBLIC int ArtmReconfigureRegularizer(int master_id, int length, const char* regularizer_config);
  DLL_PUBLIC int ArtmDisposeRegularizer(int master_id, const char* regularizer_name);

  DLL_PUBLIC int ArtmCreateDictionary(int master_id, int length, const char* dictionary_config);
  DLL_PUBLIC int ArtmReconfigureDictionary(int master_id, int length, const char* dictionary_config);
  DLL_PUBLIC int ArtmDisposeDictionary(int master_id, const char* dictionary_name);

  DLL_PUBLIC int ArtmAddBatch(int master_id, int length, const char* batch);
  DLL_PUBLIC int ArtmInvokeIteration(int master_id, int iterations_count);
  DLL_PUBLIC int ArtmInvokePhiRegularizers(int master_id);
  DLL_PUBLIC int ArtmWaitIdle(int master_id, int timeout_milliseconds);

  DLL_PUBLIC int ArtmOverwriteTopicModel(int master_id, int length, const char* topic_model);

  DLL_PUBLIC int ArtmRequestThetaMatrix(int master_id, const char* model_name);
  DLL_PUBLIC int ArtmRequestTopicModel(int master_id, const char* model_name);
  DLL_PUBLIC int ArtmRequestRegularizerState(int master_id, const char* regularizer_name);
  DLL_PUBLIC int ArtmRequestScore(int master_id, const char* model_name, const char* score_name);
  DLL_PUBLIC int ArtmRequestParseCollection(int length, const char* collection_parser_config);
  DLL_PUBLIC int ArtmRequestLoadDictionary(const char* filename);
  DLL_PUBLIC int ArtmCopyRequestResult(int length, char* address);

  DLL_PUBLIC int ArtmSaveBatch(const char* disk_path, int length, const char* batch);
  DLL_PUBLIC const char* ArtmGetLastErrorMessage();
}

#endif  // SRC_ARTM_C_INTERFACE_H_
