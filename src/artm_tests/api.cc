// Copyright 2014, Additive Regularization of Topic Models.

#include "artm_tests/api.h"

namespace artm {
namespace test {

inline char* StringAsArray(std::string* str) {
  return str->empty() ? NULL : &*str->begin();
}

template<typename ArgsT, typename FuncT>
int ArtmExecute(int master_id, const ArgsT& args, FuncT func) {
  std::string blob;
  args.SerializeToString(&blob);
  return HandleErrorCode(func(master_id, blob.size(), StringAsArray(&blob)));
}

template<typename ResultT, typename ArgsT, typename FuncT>
ResultT ArtmRequest(int master_id, const ArgsT& args, FuncT func) {
  int length = ArtmExecute(master_id, args, func);

  std::string result_blob;
  result_blob.resize(length);
  HandleErrorCode(ArtmCopyRequestResult(length, StringAsArray(&result_blob)));

  ResultT result;
  result.ParseFromString(result_blob);
  return result;
}

TopicModel Api::AttachTopicModel(const AttachModelArgs& args, Matrix* matrix) {
  GetTopicModelArgs topic_args;
  topic_args.set_model_name(args.model_name());
  topic_args.set_request_type(GetTopicModelArgs_RequestType_TopicNames);
  TopicModel topics = master_model_.GetTopicModel(topic_args);

  GetTopicModelArgs token_args;
  token_args.set_model_name(args.model_name());
  token_args.set_request_type(GetTopicModelArgs_RequestType_Tokens);
  TopicModel tokens = master_model_.GetTopicModel(token_args);

  std::string args_blob;
  args.SerializeToString(&args_blob);

  if (topics.topics_count() == 0)
    throw ArgumentOutOfRangeException("Unable to attach to topic model with zero topics");
  if (tokens.token_size() == 0)
    throw ArgumentOutOfRangeException("Unable to attach to topic model with zero tokens");

  matrix->resize(tokens.token_size(), topics.topics_count());
  int address_length = matrix->no_columns() * matrix->no_rows() * sizeof(float);
  artm::HandleErrorCode(ArtmAttachModel(master_model_.id(), args_blob.size(), args_blob.c_str(),
                        address_length, reinterpret_cast<char*>(matrix->get_data())));

  TopicModel retval;
  retval.set_topics_count(topics.topics_count());
  retval.mutable_topic_name()->CopyFrom(topics.topic_name());
  retval.mutable_class_id()->CopyFrom(tokens.class_id());
  retval.mutable_token()->CopyFrom(tokens.token());
  return retval;
}

ThetaMatrix Api::ProcessBatches(const ProcessBatchesArgs& args) {
  auto process_batches_result = ArtmRequest<ProcessBatchesResult>(master_model_.id(), args, ArtmRequestProcessBatches);
  return process_batches_result.theta_matrix();
}

int Api::AsyncProcessBatches(const ProcessBatchesArgs& args) {
  return ArtmExecute(master_model_.id(), args, ArtmAsyncProcessBatches);
}

int Api::AwaitOperation(int operation_id) {
  return ARTM_STILL_WORKING != ArtmExecute(operation_id, AwaitOperationArgs(), ArtmAwaitOperation);
}

void Api::MergeModel(const MergeModelArgs& args) {
  ArtmExecute(master_model_.id(), args, ArtmMergeModel);
}

void Api::NormalizeModel(const NormalizeModelArgs& args) {
  ArtmExecute(master_model_.id(), args, ArtmNormalizeModel);
}

void Api::RegularizeModel(const RegularizeModelArgs& args) {
  ArtmExecute(master_model_.id(), args, ArtmRegularizeModel);
}

void Api::OverwriteModel(const TopicModel& args) {
  ArtmExecute(master_model_.id(), args, ArtmOverwriteTopicModel);
}

::artm::FitOfflineMasterModelArgs Api::Initialize(const std::vector<std::shared_ptr< ::artm::Batch> >& batches,
                                                  ::artm::ImportBatchesArgs* import_batches_args,
                                                  ::artm::InitializeModelArgs* initialize_model_args) {
  ImportBatchesArgs import_args;
  for (auto& batch : batches) {
    import_args.add_batch()->CopyFrom(*batch);
    import_args.add_batch_name(batch->id());
  }
  master_model_.ImportBatches(import_args);
  if (import_batches_args != nullptr)
    import_batches_args->CopyFrom(import_args);

  ::artm::GatherDictionaryArgs gather_args;
  gather_args.mutable_batch_path()->CopyFrom(import_args.batch_name());
  gather_args.set_dictionary_target_name("dictionary");
  master_model_.GatherDictionary(gather_args);

  ::artm::InitializeModelArgs init_model_args;
  init_model_args.set_dictionary_name("dictionary");
  init_model_args.set_model_name(master_model_.config().pwt_name());
  init_model_args.mutable_topic_name()->CopyFrom(master_model_.config().topic_name());
  master_model_.InitializeModel(init_model_args);
  if (initialize_model_args != nullptr)
    initialize_model_args->CopyFrom(init_model_args);

  ::artm::FitOfflineMasterModelArgs fit_offline_args;
  fit_offline_args.mutable_batch_filename()->CopyFrom(import_args.batch_name());
  return fit_offline_args;
}

}  // namespace test
}  // namespace artm
