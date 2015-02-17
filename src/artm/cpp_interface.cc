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
    case ARTM_NETWORK_ERROR:
      throw NetworkException(GetLastErrorMessage());
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

std::shared_ptr<DictionaryConfig> LoadDictionary(const std::string& filename) {
  int length = HandleErrorCode(ArtmRequestLoadDictionary(filename.c_str()));

  std::string dictionary_blob;
  dictionary_blob.resize(length);
  HandleErrorCode(ArtmCopyRequestResult(length, StringAsArray(&dictionary_blob)));

  std::shared_ptr<DictionaryConfig> dictionary(new DictionaryConfig());
  dictionary->ParseFromString(dictionary_blob);
  return dictionary;
}

std::shared_ptr<DictionaryConfig> ParseCollection(const CollectionParserConfig& config) {
  std::string config_blob;
  config.SerializeToString(&config_blob);
  int length = HandleErrorCode(ArtmRequestParseCollection(config_blob.size(),
    StringAsArray(&config_blob)));

  std::string dictionary_blob;
  dictionary_blob.resize(length);
  HandleErrorCode(ArtmCopyRequestResult(length, StringAsArray(&dictionary_blob)));

  std::shared_ptr<DictionaryConfig> dictionary(new DictionaryConfig());
  dictionary->ParseFromString(dictionary_blob);
  return dictionary;
}

MasterComponent::MasterComponent(const MasterComponentConfig& config) : id_(0), config_(config) {
  std::string config_blob;
  config.SerializeToString(&config_blob);
  id_ = HandleErrorCode(ArtmCreateMasterComponent(
    config_blob.size(), StringAsArray(&config_blob)));
}

MasterComponent::MasterComponent(const MasterProxyConfig& config)
    : id_(0), config_(config.config()) {
  std::string config_blob;
  config.SerializeToString(&config_blob);
  id_ = HandleErrorCode(ArtmCreateMasterProxy(config_blob.size(), StringAsArray(&config_blob)));
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

std::shared_ptr<TopicModel> MasterComponent::GetTopicModel(const GetTopicModelArgs& args) {
  // Request model topics
  std::string args_blob;
  args.SerializeToString(&args_blob);
  int length = HandleErrorCode(ArtmRequestTopicModel(id(), args_blob.size(), args_blob.c_str()));
  std::string topic_model_blob;
  topic_model_blob.resize(length);
  HandleErrorCode(ArtmCopyRequestResult(length, StringAsArray(&topic_model_blob)));

  std::shared_ptr<TopicModel> topic_model(new TopicModel());
  topic_model->ParseFromString(topic_model_blob);
  return topic_model;
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

std::shared_ptr<ThetaMatrix> MasterComponent::GetThetaMatrix(const GetThetaMatrixArgs& args) {
  std::string args_blob;
  args.SerializeToString(&args_blob);
  int length = HandleErrorCode(ArtmRequestThetaMatrix(id(), args_blob.size(), args_blob.c_str()));
  std::string blob;
  blob.resize(length);
  HandleErrorCode(ArtmCopyRequestResult(length, StringAsArray(&blob)));

  std::shared_ptr<ThetaMatrix> theta_matrix(new ThetaMatrix());
  theta_matrix->ParseFromString(blob);
  return theta_matrix;
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

NodeController::NodeController(const NodeControllerConfig& config) : id_(0), config_(config) {
  std::string config_blob;
  config.SerializeToString(&config_blob);
  id_ = HandleErrorCode(ArtmCreateNodeController(config_blob.size(), StringAsArray(&config_blob)));
}

NodeController::~NodeController() {
  ArtmDisposeNodeController(id());
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

void Model::Initialize(const Dictionary& dictionary) {
  InitializeModelArgs args;
  args.set_model_name(this->name());
  args.set_dictionary_name(dictionary.name());
  std::string blob;
  args.SerializeToString(&blob);
  HandleErrorCode(ArtmInitializeModel(master_id(), blob.size(), blob.c_str()));
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

Dictionary::Dictionary(const MasterComponent& master_component, const DictionaryConfig& config)
    : master_id_(master_component.id()),
      config_(config) {
  std::string dictionary_config_blob;
  config.SerializeToString(&dictionary_config_blob);
  HandleErrorCode(ArtmCreateDictionary(master_id_, dictionary_config_blob.size(),
    StringAsArray(&dictionary_config_blob)));
}

Dictionary::~Dictionary() {
  ArtmDisposeDictionary(master_id(), config_.name().c_str());
}

void Dictionary::Reconfigure(const DictionaryConfig& config) {
  std::string dictionary_config_blob;
  config.SerializeToString(&dictionary_config_blob);
  HandleErrorCode(ArtmReconfigureDictionary(master_id(), dictionary_config_blob.size(),
    StringAsArray(&dictionary_config_blob)));
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

}  // namespace artm
