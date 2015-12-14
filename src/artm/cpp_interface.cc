// Copyright 2014, Additive Regularization of Topic Models.

#ifdef INCLUDE_STDAFX_H
#include "stdafx.h"  // NOLINT
#endif

#include <iostream>  // NOLINT

#include "artm/cpp_interface.h"

namespace artm {

inline char* StringAsArray(std::string* str) {
  return str->empty() ? NULL : &*str->begin();
}

inline std::string GetLastErrorMessage() {
  return std::string(ArtmGetLastErrorMessage());
}

inline int HandleErrorCode(int artm_error_code) {
  // All error codes are negative. Any non-negative value is a success.
  if (artm_error_code >= 0) {
    return artm_error_code;
  }

  if (artm_error_code == ARTM_STILL_WORKING) {
    return artm_error_code;
  }

  switch (artm_error_code) {
    case ARTM_INTERNAL_ERROR:
      throw InternalError(GetLastErrorMessage());
    case ARTM_ARGUMENT_OUT_OF_RANGE:
      throw ArgumentOutOfRangeException(GetLastErrorMessage());
    case ARTM_INVALID_MASTER_ID:
      throw InvalidMasterIdException(GetLastErrorMessage());
    case ARTM_CORRUPTED_MESSAGE:
      throw CorruptedMessageException(GetLastErrorMessage());
    case ARTM_INVALID_OPERATION:
      throw InvalidOperationException(GetLastErrorMessage());
    case ARTM_DISK_READ_ERROR:
      throw DiskReadException(GetLastErrorMessage());
    case ARTM_DISK_WRITE_ERROR:
      throw DiskWriteException(GetLastErrorMessage());
    default:
      throw InternalError("Unknown error code");
  }
}

void SaveBatch(const Batch& batch, const std::string& disk_path) {
  std::string config_blob;
  batch.SerializeToString(&config_blob);
  HandleErrorCode(ArtmSaveBatch(disk_path.c_str(), config_blob.size(),
    StringAsArray(&config_blob)));
}

std::shared_ptr<Batch> LoadBatch(const std::string& filename) {
  int length = HandleErrorCode(ArtmRequestLoadBatch(filename.c_str()));

  std::string message_blob;
  message_blob.resize(length);
  HandleErrorCode(ArtmCopyRequestResult(length, StringAsArray(&message_blob)));

  std::shared_ptr<Batch> message(new Batch());
  message->ParseFromString(message_blob);
  return message;
}

void ParseCollection(const CollectionParserConfig& config) {
  std::string config_blob;
  config.SerializeToString(&config_blob);
  HandleErrorCode(ArtmParseCollection(config_blob.size(), StringAsArray(&config_blob)));
}

MasterComponent::MasterComponent(const MasterComponentConfig& config) : id_(0), config_(config) {
  std::string config_blob;
  config.SerializeToString(&config_blob);
  id_ = HandleErrorCode(ArtmCreateMasterComponent(
    config_blob.size(), StringAsArray(&config_blob)));
}

MasterComponent::MasterComponent(const MasterComponent& rhs) : id_(0), config_(rhs.config_) {
  ::artm::DuplicateMasterComponentArgs args;
  std::string config_blob;
  args.SerializeToString(&config_blob);
  id_ = HandleErrorCode(ArtmDuplicateMasterComponent(rhs.id_, config_blob.size(), StringAsArray(&config_blob)));
}

MasterComponent::~MasterComponent() {
  ArtmDisposeMasterComponent(id());
}

void MasterComponent::Reconfigure(const MasterComponentConfig& config) {
  std::string config_blob;
  config.SerializeToString(&config_blob);
  HandleErrorCode(ArtmReconfigureMasterComponent(id(), config_blob.size(), StringAsArray(&config_blob)));
  config_.CopyFrom(config);
}

std::shared_ptr<TopicModel> MasterComponent::GetTopicModel(const std::string& model_name) {
  ::artm::GetTopicModelArgs args;
  args.set_model_name(model_name);
  return GetTopicModel(args);
}

std::shared_ptr<TopicModel> MasterComponent::GetTopicModel(const std::string& model_name, Matrix* matrix) {
  ::artm::GetTopicModelArgs args;
  args.set_model_name(model_name);
  return GetTopicModel(args, matrix);
}

std::shared_ptr<TopicModel> MasterComponent::GetTopicModel(const GetTopicModelArgs& args) {
  return GetTopicModel(args, nullptr);
}

std::shared_ptr<TopicModel> MasterComponent::GetTopicModel(const GetTopicModelArgs& args, Matrix* matrix) {
  std::string args_blob;
  args.SerializeToString(&args_blob);
  auto func = (matrix == nullptr) ? ArtmRequestTopicModel : ArtmRequestTopicModelExternal;
  int length = HandleErrorCode(func(id(), args_blob.size(), args_blob.c_str()));
  std::string topic_model_blob;
  topic_model_blob.resize(length);
  HandleErrorCode(ArtmCopyRequestResult(length, StringAsArray(&topic_model_blob)));

  std::shared_ptr<TopicModel> retval(new TopicModel());
  retval->ParseFromString(topic_model_blob);

  if (matrix == nullptr)
    return retval;

  matrix->resize(retval->token_size(), retval->topics_count());

  CopyRequestResultArgs copy_request_args;
  copy_request_args.set_request_type(CopyRequestResultArgs_RequestType_GetModelSecondPass);

  args_blob.clear();
  copy_request_args.SerializeToString(&args_blob);

  length = sizeof(float) * matrix->no_columns() * matrix->no_rows();
  HandleErrorCode(ArtmCopyRequestResultEx(length, reinterpret_cast<char*>(matrix->get_data()),
    args_blob.size(), args_blob.c_str()));
  return retval;
}

std::shared_ptr<Matrix> MasterComponent::AttachTopicModel(const std::string& model_name) {
  return AttachTopicModel(model_name, nullptr);
}

std::shared_ptr<Matrix> MasterComponent::AttachTopicModel(const std::string& model_name, TopicModel* topic_model) {
  GetTopicModelArgs topic_args;
  topic_args.set_model_name(model_name);
  topic_args.set_request_type(GetTopicModelArgs_RequestType_TopicNames);
  std::shared_ptr<TopicModel> topics = GetTopicModel(topic_args);

  GetTopicModelArgs token_args;
  token_args.set_model_name(model_name);
  token_args.set_request_type(GetTopicModelArgs_RequestType_Tokens);
  std::shared_ptr<TopicModel> tokens = GetTopicModel(token_args);

  AttachModelArgs attach_model_args;
  attach_model_args.set_model_name(model_name);
  std::string args_blob;
  attach_model_args.SerializeToString(&args_blob);

  if (topics->topics_count() == 0)
    throw ArgumentOutOfRangeException("Unable to attach to topic model with zero topics");
  if (tokens->token_size() == 0)
    throw ArgumentOutOfRangeException("Unable to attach to topic model with zero tokens");

  std::shared_ptr<Matrix> matrix = std::make_shared<Matrix>(tokens->token_size(), topics->topics_count());
  int address_length = matrix->no_columns() * matrix->no_rows() * sizeof(float);
  HandleErrorCode(ArtmAttachModel(id(), args_blob.size(), args_blob.c_str(),
                  address_length, reinterpret_cast<char*>(matrix->get_data())));

  if (topic_model != nullptr) {
    topic_model->set_topics_count(topics->topics_count());
    topic_model->mutable_topic_name()->CopyFrom(topics->topic_name());
    topic_model->mutable_class_id()->CopyFrom(tokens->class_id());
    topic_model->mutable_token()->CopyFrom(tokens->token());
  }

  return matrix;
}

std::shared_ptr<RegularizerInternalState> MasterComponent::GetRegularizerState(
  const std::string& regularizer_name) {
  int length = HandleErrorCode(ArtmRequestRegularizerState(id(), regularizer_name.c_str()));
  std::string state_blob;
  state_blob.resize(length);
  HandleErrorCode(ArtmCopyRequestResult(length, StringAsArray(&state_blob)));

  std::shared_ptr<RegularizerInternalState> regularizer_state(new RegularizerInternalState());
  regularizer_state->ParseFromString(state_blob);
  return regularizer_state;
}

std::shared_ptr<ThetaMatrix> MasterComponent::GetThetaMatrix(const std::string& model_name) {
  ::artm::GetThetaMatrixArgs args;
  args.set_model_name(model_name);
  return GetThetaMatrix(args);
}

std::shared_ptr<ThetaMatrix> MasterComponent::GetThetaMatrix(const std::string& model_name, Matrix* matrix) {
  ::artm::GetThetaMatrixArgs args;
  args.set_model_name(model_name);
  return GetThetaMatrix(args, matrix);
}

std::shared_ptr<ThetaMatrix> MasterComponent::GetThetaMatrix(const std::string& model_name,
                                                             const ::artm::Batch& batch) {
  ::artm::GetThetaMatrixArgs args;
  args.set_model_name(model_name);
  args.mutable_batch()->CopyFrom(batch);
  return GetThetaMatrix(args);
}

std::shared_ptr<ThetaMatrix> MasterComponent::GetThetaMatrix(const GetThetaMatrixArgs& args) {
  return GetThetaMatrix(args, nullptr);
}

std::shared_ptr<ThetaMatrix> MasterComponent::GetThetaMatrix(const GetThetaMatrixArgs& args, Matrix* matrix) {
  std::string args_blob;
  args.SerializeToString(&args_blob);
  auto func = (matrix == nullptr) ? ArtmRequestThetaMatrix : ArtmRequestThetaMatrixExternal;
  int length = HandleErrorCode(func(id(), args_blob.size(), args_blob.c_str()));
  std::string blob;
  blob.resize(length);
  HandleErrorCode(ArtmCopyRequestResult(length, StringAsArray(&blob)));

  std::shared_ptr<ThetaMatrix> retval(new ThetaMatrix());
  retval->ParseFromString(blob);

  if (matrix == nullptr)
    return retval;

  matrix->resize(retval->item_id_size(), retval->topics_count());

  CopyRequestResultArgs copy_request_args;
  copy_request_args.set_request_type(CopyRequestResultArgs_RequestType_GetThetaSecondPass);
  args_blob.clear();
  copy_request_args.SerializeToString(&args_blob);

  length = sizeof(float) * matrix->no_columns() * matrix->no_rows();
  HandleErrorCode(ArtmCopyRequestResultEx(length, reinterpret_cast<char*>(matrix->get_data()),
                                          args_blob.size(), args_blob.c_str()));
  return retval;
}

std::shared_ptr<ScoreData> MasterComponent::GetScore(const GetScoreValueArgs& args) {
  std::string args_blob;
  args.SerializeToString(&args_blob);
  int length = HandleErrorCode(ArtmRequestScore(id(), args_blob.size(), args_blob.c_str()));
  std::string blob;
  blob.resize(length);
  HandleErrorCode(ArtmCopyRequestResult(length, StringAsArray(&blob)));

  std::shared_ptr<ScoreData> score_data(new ScoreData());
  score_data->ParseFromString(blob);
  return score_data;
}

std::shared_ptr<MasterComponentInfo> MasterComponent::info() const {
  GetMasterComponentInfoArgs args;
  std::string args_blob;
  args.SerializeToString(&args_blob);
  int length = HandleErrorCode(ArtmRequestMasterComponentInfo(id(), args_blob.size(), args_blob.c_str()));
  std::string blob;
  blob.resize(length);
  HandleErrorCode(ArtmCopyRequestResult(length, StringAsArray(&blob)));

  std::shared_ptr<MasterComponentInfo> master_component_info(new MasterComponentInfo());
  master_component_info->ParseFromString(blob);
  return master_component_info;
}

Model::Model(const MasterComponent& master_component, const ModelConfig& config)
    : master_id_(master_component.id()),
      config_(config) {
  if (!config_.has_name())
    throw ArgumentOutOfRangeException("model_config.has_name()==false");

  std::string model_config_blob;
  config_.SerializeToString(&model_config_blob);
  HandleErrorCode(ArtmCreateModel(master_id_, model_config_blob.size(),
    StringAsArray(&model_config_blob)));
}

Model::~Model() {
  ArtmDisposeModel(master_id(), name().c_str());
}

void Model::Reconfigure(const ModelConfig& config) {
  if (name() != config.name())
    throw InvalidOperationException("Changing model name is not allowed");
  std::string model_config_blob;
  config.SerializeToString(&model_config_blob);
  HandleErrorCode(ArtmReconfigureModel(master_id(), model_config_blob.size(),
    StringAsArray(&model_config_blob)));
  config_.CopyFrom(config);
}

void Model::Overwrite(const TopicModel& topic_model) {
  Overwrite(topic_model, true);
}

void Model::Overwrite(const TopicModel& topic_model, bool commit) {
  std::string blob;
  TopicModel topic_model_copy(topic_model);
  topic_model_copy.set_name(name());
  topic_model_copy.SerializeToString(&blob);
  HandleErrorCode(ArtmOverwriteTopicModel(master_id(), blob.size(), StringAsArray(&blob)));
  if (commit) {
    WaitIdleArgs args;
    std::string args_blob;
    args.set_timeout_milliseconds(-1);
    args.SerializeToString(&args_blob);
    HandleErrorCode(ArtmWaitIdle(master_id(), args_blob.size(), StringAsArray(&args_blob)));
    Synchronize(0.0, 1.0, false);
  }
}

void Model::Enable() {
  ModelConfig config_copy_(config_);
  config_copy_.set_enabled(true);
  Reconfigure(config_copy_);
}

void Model::Disable() {
  ModelConfig config_copy_(config_);
  config_copy_.set_enabled(false);
  Reconfigure(config_copy_);
}

void Model::Initialize(const std::string& dictionary_name) {
  InitializeModelArgs args;
  args.set_model_name(this->name());
  args.set_dictionary_name(dictionary_name);
  std::string blob;
  args.SerializeToString(&blob);
  HandleErrorCode(ArtmInitializeModel(master_id(), blob.size(), blob.c_str()));
}

void Model::Export(const std::string& file_name) {
  ExportModelArgs args;
  args.set_model_name(this->name());
  args.set_file_name(file_name);
  std::string blob;
  args.SerializeToString(&blob);
  HandleErrorCode(ArtmExportModel(master_id(), blob.size(), blob.c_str()));
}

void Model::Import(const std::string& file_name) {
  ImportModelArgs args;
  args.set_model_name(this->name());
  args.set_file_name(file_name);
  std::string blob;
  args.SerializeToString(&blob);
  HandleErrorCode(ArtmImportModel(master_id(), blob.size(), blob.c_str()));
}

void Model::Synchronize(double decay) {
  Synchronize(decay, 1.0, true);
}

void Model::Synchronize(double decay, double apply, bool invoke_regularizers) {
  SynchronizeModelArgs args;
  args.set_model_name(this->name());
  args.set_decay_weight(static_cast<float>(decay));
  args.set_apply_weight(static_cast<float>(apply));
  args.set_invoke_regularizers(invoke_regularizers);
  Synchronize(args);
}

void Model::Synchronize(const SynchronizeModelArgs& sync_args) {
  SynchronizeModelArgs args(sync_args);  // create a copy to set the name.
  args.set_model_name(this->name());
  std::string blob;
  args.SerializeToString(&blob);
  HandleErrorCode(ArtmSynchronizeModel(master_id(), blob.size(), blob.c_str()));
}

Regularizer::Regularizer(const MasterComponent& master_component, const RegularizerConfig& config)
    : master_id_(master_component.id()),
      config_(config) {
  std::string regularizer_config_blob;
  config.SerializeToString(&regularizer_config_blob);
  HandleErrorCode(ArtmCreateRegularizer(master_id_, regularizer_config_blob.size(),
    StringAsArray(&regularizer_config_blob)));
}

Regularizer::~Regularizer() {
  ArtmDisposeRegularizer(master_id(), config_.name().c_str());
}

void Regularizer::Reconfigure(const RegularizerConfig& config) {
  std::string regularizer_config_blob;
  config.SerializeToString(&regularizer_config_blob);
  HandleErrorCode(ArtmReconfigureRegularizer(master_id(), regularizer_config_blob.size(),
    StringAsArray(&regularizer_config_blob)));
  config_.CopyFrom(config);
}

bool MasterComponent::AddBatch(const Batch& batch) {
  return AddBatch(batch, /*bool reset_scores=*/ false);
}

bool MasterComponent::AddBatch(const Batch& batch, bool reset_scores) {
  AddBatchArgs args;
  args.mutable_batch()->CopyFrom(batch);
  args.set_reset_scores(reset_scores);
  return AddBatch(args);
}

bool MasterComponent::AddBatch(const AddBatchArgs& args) {
  std::string args_blob;
  args.SerializeToString(&args_blob);
  int result = ArtmAddBatch(id(), args_blob.size(), StringAsArray(&args_blob));
  if (result == ARTM_STILL_WORKING) {
    return false;
  } else {
    HandleErrorCode(result);
    return true;
  }
}

void MasterComponent::InvokeIteration(int iterations_count) {
  InvokeIterationArgs args;
  args.set_iterations_count(iterations_count);
  InvokeIteration(args);
}

void MasterComponent::InvokeIteration(const InvokeIterationArgs& args) {
  std::string args_blob;
  args.SerializeToString(&args_blob);
  HandleErrorCode(ArtmInvokeIteration(id(), args_blob.size(), StringAsArray(&args_blob)));
}

bool MasterComponent::WaitIdle(int timeout) {
  WaitIdleArgs args;
  args.set_timeout_milliseconds(timeout);
  return WaitIdle(args);
}

bool MasterComponent::WaitIdle(const WaitIdleArgs& args) {
  std::string args_blob;
  args.SerializeToString(&args_blob);
  int result = ArtmWaitIdle(id(), args_blob.size(), StringAsArray(&args_blob));
  if (result == ARTM_STILL_WORKING) {
    return false;
  } else {
    HandleErrorCode(result);
    return true;
  }
}

void MasterComponent::AddStream(const Stream& stream) {
  Stream* s = config_.add_stream();
  s->CopyFrom(stream);
  Reconfigure(config_);
}

void MasterComponent::RemoveStream(std::string stream_name) {
  MasterComponentConfig new_config(config_);
  new_config.mutable_stream()->Clear();

  for (int stream_index = 0;
       stream_index < config_.stream_size();
       ++stream_index) {
    if (config_.stream(stream_index).name() != stream_name) {
      Stream* s = new_config.add_stream();
      s->CopyFrom(config_.stream(stream_index));
    }
  }

  Reconfigure(new_config);
}

void MasterComponent::ExportModel(const ExportModelArgs& args) {
  std::string blob;
  args.SerializeToString(&blob);
  HandleErrorCode(ArtmExportModel(id_, blob.size(), blob.c_str()));
}

void MasterComponent::ImportModel(const ImportModelArgs& args) {
  std::string blob;
  args.SerializeToString(&blob);
  HandleErrorCode(ArtmImportModel(id_, blob.size(), blob.c_str()));
}

void MasterComponent::DisposeModel(const std::string& model_name) {
  ArtmDisposeModel(id(), model_name.c_str());
}

void MasterComponent::CreateDictionary(const DictionaryData& args) {
  std::string blob;
  args.SerializeToString(&blob);
  HandleErrorCode(ArtmCreateDictionary(id_, blob.size(), blob.c_str()));
}

void MasterComponent::DisposeDictionary(const std::string& dictionary_name) {
  HandleErrorCode(ArtmDisposeDictionary(id_, dictionary_name.c_str()));
}

void MasterComponent::ImportDictionary(const ImportDictionaryArgs& args) {
  std::string blob;
  args.SerializeToString(&blob);
  HandleErrorCode(ArtmImportDictionary(id_, blob.size(), blob.c_str()));
}

void MasterComponent::GatherDictionary(const GatherDictionaryArgs& args) {
  std::string blob;
  args.SerializeToString(&blob);
  HandleErrorCode(ArtmGatherDictionary(id_, blob.size(), blob.c_str()));
}

void MasterComponent::FilterDictionary(const FilterDictionaryArgs& args) {
  std::string blob;
  args.SerializeToString(&blob);
  HandleErrorCode(ArtmFilterDictionary(id_, blob.size(), blob.c_str()));
}

std::shared_ptr<DictionaryData> MasterComponent::GetDictionary(const std::string& dictionary_name) {
  artm::GetDictionaryArgs args;
  args.set_dictionary_name(dictionary_name);
  std::string args_blob;
  args.SerializeToString(&args_blob);
  args.set_dictionary_name(dictionary_name);
  int length = HandleErrorCode(ArtmRequestDictionary(id(), args_blob.size(), args_blob.c_str()));
  std::string state_blob;
  state_blob.resize(length);
  HandleErrorCode(ArtmCopyRequestResult(length, StringAsArray(&state_blob)));

  std::shared_ptr<DictionaryData> result(new DictionaryData());
  result->ParseFromString(state_blob);
  return result;
}

std::shared_ptr<ProcessBatchesResultObject> MasterComponent::ProcessBatches(const ProcessBatchesArgs& args) {
  std::string args_blob;
  args.SerializeToString(&args_blob);
  int length = HandleErrorCode(ArtmRequestProcessBatches(id(), args_blob.size(), args_blob.c_str()));
  std::string process_batches_result_blob;
  process_batches_result_blob.resize(length);
  HandleErrorCode(ArtmCopyRequestResult(length, StringAsArray(&process_batches_result_blob)));

  ProcessBatchesResult process_batches_result;
  process_batches_result.ParseFromString(process_batches_result_blob);

  std::shared_ptr<ProcessBatchesResultObject> retval(new ProcessBatchesResultObject(process_batches_result));
  return retval;
}

int MasterComponent::AsyncProcessBatches(const ProcessBatchesArgs& args) {
  std::string args_blob;
  args.SerializeToString(&args_blob);
  return HandleErrorCode(ArtmAsyncProcessBatches(id(), args_blob.size(), args_blob.c_str()));
}

int MasterComponent::AwaitOperation(int operation_id) {
  ::artm::AwaitOperationArgs args;
  // args.set_timeout_milliseconds(timeout_milliseconds);
  std::string args_blob;
  args.SerializeToString(&args_blob);
  int code = HandleErrorCode(ArtmAwaitOperation(operation_id, args_blob.size(), StringAsArray(&args_blob)));
  if (code == ARTM_STILL_WORKING)
    return false;
  return true;
}

void MasterComponent::MergeModel(const MergeModelArgs& args) {
  std::string args_blob;
  args.SerializeToString(&args_blob);
  HandleErrorCode(ArtmMergeModel(id(), args_blob.size(), StringAsArray(&args_blob)));
}

void MasterComponent::NormalizeModel(const NormalizeModelArgs& args) {
  std::string args_blob;
  args.SerializeToString(&args_blob);
  HandleErrorCode(ArtmNormalizeModel(id(), args_blob.size(), StringAsArray(&args_blob)));
}

void MasterComponent::RegularizeModel(const RegularizeModelArgs& args) {
  std::string args_blob;
  args.SerializeToString(&args_blob);
  HandleErrorCode(ArtmRegularizeModel(id(), args_blob.size(), StringAsArray(&args_blob)));
}

void MasterComponent::InitializeModel(const InitializeModelArgs& args) {
  std::string blob;
  args.SerializeToString(&blob);
  HandleErrorCode(ArtmInitializeModel(id(), blob.size(), blob.c_str()));
}

void MasterComponent::ImportBatches(const ImportBatchesArgs& args) {
  std::string blob;
  args.SerializeToString(&blob);
  HandleErrorCode(ArtmImportBatches(id(), blob.size(), blob.c_str()));
}

void MasterComponent::DisposeBatches(const DisposeBatchesArgs& args) {
  std::string blob;
  args.SerializeToString(&blob);
  HandleErrorCode(ArtmDisposeBatches(id(), blob.size(), blob.c_str()));
}

}  // namespace artm
