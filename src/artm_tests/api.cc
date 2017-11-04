// Copyright 2017, Additive Regularization of Topic Models.

#include "artm_tests/api.h"

#include "google/protobuf/util/json_util.h"

namespace artm {
namespace test {

inline char* StringAsArray(std::string* str) {
  return str->empty() ? NULL : &*str->begin();
}

static void ParseMessageFromString(const std::string& string, google::protobuf::Message* message) {
  if (ArtmProtobufMessageFormatIsJson()) {
    ::google::protobuf::util::JsonStringToMessage(string, message);
  } else {
    message->ParseFromString(string);
  }
}

static void SerializeMessageToString(const google::protobuf::Message& message, std::string* output) {
  if (ArtmProtobufMessageFormatIsJson()) {
    ::google::protobuf::util::MessageToJsonString(message, output);
  } else {
    output->clear();
    message.SerializeToString(output);
  }
}

template<typename ArgsT, typename FuncT>
int ArtmExecute(int master_id, const ArgsT& args, FuncT func) {
  std::string blob;
  SerializeMessageToString(args, &blob);
  return HandleErrorCode(func(master_id, blob.size(), StringAsArray(&blob)));
}

template<typename ResultT, typename ArgsT, typename FuncT>
ResultT ArtmRequest(int master_id, const ArgsT& args, FuncT func) {
  int length = ArtmExecute(master_id, args, func);

  std::string result_blob;
  result_blob.resize(length);
  HandleErrorCode(ArtmCopyRequestedMessage(length, StringAsArray(&result_blob)));

  ResultT result;
  ParseMessageFromString(result_blob, &result);
  return result;
}

TopicModel Api::AttachTopicModel(const AttachModelArgs& args, Matrix* matrix) {
  GetTopicModelArgs topic_args;
  topic_args.set_model_name(args.model_name());
  topic_args.set_matrix_layout(MatrixLayout_Sparse);
  topic_args.set_eps(1.001f);  // hack-hack to return no entries
  TopicModel retval = master_model_.GetTopicModel(topic_args);

  std::string args_blob;
  SerializeMessageToString(args, &args_blob);

  if (retval.num_topics() == 0) {
    throw ArgumentOutOfRangeException("Unable to attach to topic model with zero topics");
  }
  if (retval.token_size() == 0) {
    throw ArgumentOutOfRangeException("Unable to attach to topic model with zero tokens");
  }

  matrix->resize(retval.token_size(), retval.num_topics());
  int address_length = matrix->no_columns() * matrix->no_rows() * sizeof(float);
  artm::HandleErrorCode(ArtmAttachModel(master_model_.id(), args_blob.size(), args_blob.c_str(),
                        address_length, reinterpret_cast<char*>(matrix->get_data())));

  return retval;
}

int Api::ClearThetaCache(const ClearThetaCacheArgs& args) {
  return ArtmExecute(master_model_.id(), args, ArtmClearThetaCache);
}

int Api::ClearScoreCache(const ClearScoreCacheArgs& args) {
  return ArtmExecute(master_model_.id(), args, ArtmClearScoreCache);
}

int Api::ClearScoreArrayCache(const ClearScoreArrayCacheArgs& args) {
  return ArtmExecute(master_model_.id(), args, ArtmClearScoreArrayCache);
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

int Api::Duplicate(const DuplicateMasterComponentArgs& args) {
  return ArtmExecute(master_model_.id(), args, ArtmDuplicateMasterComponent);
}

::artm::FitOfflineMasterModelArgs Api::Initialize(const std::vector<std::shared_ptr< ::artm::Batch> >& batches,
                                                  ::artm::ImportBatchesArgs* import_batches_args,
                                                  ::artm::InitializeModelArgs* initialize_model_args,
                                                  const ::artm::DictionaryData* dictionary_data) {
  ImportBatchesArgs import_args;
  for (auto& batch : batches) {
    import_args.add_batch()->CopyFrom(*batch);
  }
  master_model_.ImportBatches(import_args);
  if (import_batches_args != nullptr) {
    import_batches_args->CopyFrom(import_args);
  }

  ::artm::FitOfflineMasterModelArgs fit_offline_args;
  for (auto& batch : import_args.batch()) {
    fit_offline_args.add_batch_filename(batch.id());
  }

  if (dictionary_data == nullptr) {
    ::artm::GatherDictionaryArgs gather_args;
    gather_args.mutable_batch_path()->CopyFrom(fit_offline_args.batch_filename());
    gather_args.set_dictionary_target_name("dictionary");
    master_model_.GatherDictionary(gather_args);
  } else {
    const_cast< ::artm::DictionaryData*>(dictionary_data)->set_name("dictionary");
    master_model_.CreateDictionary(*dictionary_data);
  }

  ::artm::InitializeModelArgs init_model_args;
  init_model_args.set_dictionary_name("dictionary");
  init_model_args.set_model_name(master_model_.config().pwt_name());
  init_model_args.mutable_topic_name()->CopyFrom(master_model_.config().topic_name());
  master_model_.InitializeModel(init_model_args);
  if (initialize_model_args != nullptr) {
    initialize_model_args->CopyFrom(init_model_args);
  }

  return fit_offline_args;
}

}  // namespace test
}  // namespace artm
