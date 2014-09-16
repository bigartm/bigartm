// Copyright 2014, Additive Regularization of Topic Models.

#include <iostream>  // NOLINT

#include "artm/cpp_interface.h"

#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid.hpp"
#include "boost/uuid/uuid_generators.hpp"
#include "boost/uuid/uuid_io.hpp"

#include "glog/logging.h"

#include "artm/core/protobuf_helpers.h"

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

std::shared_ptr<DictionaryConfig> LoadDictionary(const std::string& disk_path) {
  int length = HandleErrorCode(ArtmRequestLoadDictionary(disk_path.c_str()));

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

std::shared_ptr<TopicModel> MasterComponent::GetTopicModel(const Model& model) {
  // Request model topics
  int length = HandleErrorCode(ArtmRequestTopicModel(id(), model.name().c_str()));
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

std::shared_ptr<ThetaMatrix> MasterComponent::GetThetaMatrix(const Model& model) {
  int length = HandleErrorCode(ArtmRequestThetaMatrix(id(), model.name().c_str()));
  std::string blob;
  blob.resize(length);
  HandleErrorCode(ArtmCopyRequestResult(length, StringAsArray(&blob)));

  std::shared_ptr<ThetaMatrix> theta_matrix(new ThetaMatrix());
  theta_matrix->ParseFromString(blob);
  return theta_matrix;
}

std::shared_ptr<ScoreData> MasterComponent::GetScore(const Model& model,
                                                     const std::string& score_name) {
  int length = HandleErrorCode(ArtmRequestScore(id(), model.name().c_str(), score_name.c_str()));
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
  if (!config_.has_name()) {
    config_.set_name(boost::lexical_cast<std::string>(boost::uuids::random_generator()()));
  }

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
  std::string blob;
  topic_model.SerializeToString(&blob);
  HandleErrorCode(ArtmOverwriteTopicModel(master_id(), blob.size(), StringAsArray(&blob)));
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

void Model::InvokePhiRegularizers() {
  HandleErrorCode(ArtmInvokePhiRegularizers(master_id()));
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

void MasterComponent::AddBatch(const Batch& batch) {
  std::string batch_blob;
  batch.SerializeToString(&batch_blob);
  HandleErrorCode(ArtmAddBatch(id(), batch_blob.size(), StringAsArray(&batch_blob)));
}

void MasterComponent::InvokeIteration(int iterations_count) {
  HandleErrorCode(ArtmInvokeIteration(id(), iterations_count));
}

bool MasterComponent::WaitIdle(int timeout) {
  int result = ArtmWaitIdle(id(), timeout);
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
