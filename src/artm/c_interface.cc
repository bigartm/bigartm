// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/c_interface.h"

#include <string>

#include "boost/thread/tss.hpp"

#include "glog/logging.h"

#include "rpcz/rpc.hpp"

#include "artm/messages.pb.h"
#include "artm/score_calculator_interface.h"
#include "artm/core/common.h"
#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/master_component.h"
#include "artm/core/master_proxy.h"
#include "artm/core/node_controller.h"
#include "artm/core/collection_parser.h"

// Never use the following variables explicitly (only through the corresponding methods).
// It might be good idea to make them a private members of a new singleton class.
static boost::thread_specific_ptr<std::string> last_message_;
static boost::thread_specific_ptr<std::string> last_error_;

static std::string* last_message() {
  if (last_message_.get() == nullptr) {
    last_message_.reset(new std::string());
  }

  return last_message_.get();
}

static void set_last_error(const std::string& error) {
  if (last_error_.get() == nullptr) {
    last_error_.reset(new std::string());
  }

  last_error_->assign(error);
}

static void EnableLogging() {
  static bool logging_enabled = false;
  if (!logging_enabled) {
    logging_enabled = true;
    FLAGS_log_dir = ".";
    FLAGS_logbufsecs = 0;
    ::google::InitGoogleLogging(".");
  }
}

static std::shared_ptr<::artm::core::MasterInterface> master_component(int master_id) {
  auto master_component = artm::core::MasterComponentManager::singleton().Get(master_id);
  if (master_component == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InvalidMasterIdException(boost::lexical_cast<std::string>(master_id)));
  }

  return master_component;
}

static char* StringAsArray(std::string* str) {
  return str->empty() ? NULL : &*str->begin();
}

static void ParseFromArray(const char* buffer, int length, google::protobuf::Message* message) {
  if (!message->ParseFromArray(buffer, length))
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException("Unable to parse the message"));
}

// =========================================================================
// Common routines
// =========================================================================

int ArtmCopyRequestResult(int length, char* address) {
  if (length != static_cast<int>(last_message()->size())) {
    set_last_error("ArtmCopyRequestResult() called with invalid 'length' parameter.");
    return ARTM_INVALID_OPERATION;
  }

  memcpy(address, StringAsArray(last_message()), length);
  return ARTM_SUCCESS;
}

const char* ArtmGetLastErrorMessage() {
  if (last_error_.get() == nullptr) {
    return nullptr;
  }

  return last_error_->c_str();
}

int ArtmSaveBatch(const char* disk_path, int length, const char* batch) {
  try {
    EnableLogging();
    artm::Batch batch_object;
    ParseFromArray(batch, length, &batch_object);

    artm::Batch compacted_batch;
    artm::core::BatchHelpers::CompactBatch(batch_object, &compacted_batch);
    artm::core::BatchHelpers::PopulateClassId(&compacted_batch);
    artm::core::BatchHelpers::SaveBatch(compacted_batch, std::string(disk_path));
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmAddBatch(int master_id, int length, const char* batch) {
  try {
    artm::Batch batch_object;
    ParseFromArray(batch, length, &batch_object);
    master_component(master_id)->AddBatch(batch_object);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmInvokeIteration(int master_id, int iterations_count) {
  try {
    master_component(master_id)->InvokeIteration(iterations_count);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmWaitIdle(int master_id, int timeout_milliseconds) {
  try {
    bool result = master_component(master_id)->WaitIdle(timeout_milliseconds);

    if (result) {
      return ARTM_SUCCESS;
    } else {
      set_last_error("Artm is still processing the collection. Call ArtmWaitIdle() later.");
      return ARTM_STILL_WORKING;
    }
  } CATCH_EXCEPTIONS;
}

// =========================================================================
// MasterComponent / MasterProxy
// =========================================================================

int ArtmCreateMasterProxy(int length, const char* master_proxy_config) {
  try {
    EnableLogging();

    artm::MasterProxyConfig config;
    ParseFromArray(master_proxy_config, length, &config);
    auto& mcm = artm::core::MasterComponentManager::singleton();
    int retval = mcm.Create<::artm::core::MasterProxy, ::artm::MasterProxyConfig>(config);
    assert(retval > 0);
    return retval;
  } CATCH_EXCEPTIONS;
}

int ArtmCreateMasterComponent(int length, const char* master_component_config) {
  try {
    EnableLogging();

    artm::MasterComponentConfig config;
    ParseFromArray(master_component_config, length, &config);
    auto& mcm = artm::core::MasterComponentManager::singleton();
    int retval = mcm.Create<::artm::core::MasterComponent, ::artm::MasterComponentConfig>(config);
    assert(retval > 0);
    return retval;
  } CATCH_EXCEPTIONS;
}

int ArtmCreateModel(int master_id, int length, const char* model_config) {
  return ArtmReconfigureModel(master_id, length, model_config);
}

int ArtmReconfigureMasterComponent(int master_id, int length, const char* master_component_config) {
  try {
    artm::MasterComponentConfig config;
    ParseFromArray(master_component_config, length, &config);
    master_component(master_id)->Reconfigure(config);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmReconfigureModel(int master_id, int length, const char* model_config) {
  try {
    artm::ModelConfig config;
    ParseFromArray(model_config, length, &config);
    master_component(master_id)->CreateOrReconfigureModel(config);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmRequestThetaMatrix(int master_id, const char* model_name) {
  try {
    artm::ThetaMatrix theta_matrix;
    master_component(master_id)->RequestThetaMatrix(model_name, &theta_matrix);
    theta_matrix.SerializeToString(last_message());
    return last_message()->size();
  } CATCH_EXCEPTIONS;
}

int ArtmRequestTopicModel(int master_id, const char* model_name) {
  try {
    artm::TopicModel topic_model;
    master_component(master_id)->RequestTopicModel(model_name, &topic_model);
    topic_model.SerializeToString(last_message());
    return last_message()->size();
  } CATCH_EXCEPTIONS;
}

int ArtmRequestRegularizerState(int master_id, const char* regularizer_name) {
  try {
    artm::RegularizerInternalState regularizer_state;
    master_component(master_id)->RequestRegularizerState(regularizer_name, &regularizer_state);
    regularizer_state.SerializeToString(last_message());
    return last_message()->size();
  } CATCH_EXCEPTIONS;
}

int ArtmRequestScore(int master_id, const char* model_name, const char* score_name) {
  try {
    ::artm::ScoreData score_data;
    master_component(master_id)->RequestScore(model_name, score_name, &score_data);
    score_data.SerializeToString(last_message());
    return last_message()->size();
  } CATCH_EXCEPTIONS;
}

int ArtmOverwriteTopicModel(int master_id, int length, const char* topic_model) {
  try {
    artm::TopicModel topic_model_object;
    ParseFromArray(topic_model, length, &topic_model_object);
    master_component(master_id)->OverwriteTopicModel(topic_model_object);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmDisposeMasterComponent(int master_id) {
  try {
    artm::core::MasterComponentManager::singleton().Erase(master_id);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmCreateNodeController(int length, const char* node_controller_config) {
  try {
    EnableLogging();

    artm::NodeControllerConfig config;
    ParseFromArray(node_controller_config, length, &config);
    auto& ncm = artm::core::NodeControllerManager::singleton();
    int retval = ncm.Create<::artm::core::NodeController, ::artm::NodeControllerConfig>(config);
    assert(retval > 0);
    return retval;
  } CATCH_EXCEPTIONS;
}

int ArtmDisposeNodeController(int node_controller_id) {
  try {
    artm::core::NodeControllerManager::singleton().Erase(node_controller_id);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmDisposeModel(int master_id, const char* model_name) {
  try {
    master_component(master_id)->DisposeModel(model_name);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmCreateRegularizer(int master_id, int length, const char* regularizer_config) {
  return ArtmReconfigureRegularizer(master_id, length, regularizer_config);
}

int ArtmReconfigureRegularizer(int master_id, int length, const char* regularizer_config) {
  try {
    ::artm::RegularizerConfig config;
    ParseFromArray(regularizer_config, length, &config);
    master_component(master_id)->CreateOrReconfigureRegularizer(config);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmDisposeRegularizer(int master_id, const char* regularizer_name) {
  try {
    master_component(master_id)->DisposeRegularizer(regularizer_name);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmInvokePhiRegularizers(int master_id) {
  try {
    master_component(master_id)->InvokePhiRegularizers();
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmCreateDictionary(int master_id, int length, const char* dictionary_config) {
  return ArtmReconfigureDictionary(master_id, length, dictionary_config);
}

int ArtmReconfigureDictionary(int master_id, int length, const char* dictionary_config) {
  try {
    artm::DictionaryConfig config;
    ParseFromArray(dictionary_config, length, &config);
    master_component(master_id)->CreateOrReconfigureDictionary(config);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmDisposeDictionary(int master_id, const char* dictionary_name) {
  try {
    master_component(master_id)->DisposeDictionary(dictionary_name);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmRequestParseCollection(int length, const char* collection_parser_config) {
  try {
    EnableLogging();
    artm::CollectionParserConfig config;
    ParseFromArray(collection_parser_config, length, &config);
    ::artm::core::CollectionParser collection_parser(config);
    std::shared_ptr<::artm::DictionaryConfig> dictionary = collection_parser.Parse();
    dictionary->SerializeToString(last_message());
    return last_message()->size();
  } CATCH_EXCEPTIONS;
}

int ArtmRequestLoadDictionary(const char* filename) {
  try {
    EnableLogging();
    auto dictionary = std::make_shared<::artm::DictionaryConfig>();
    ::artm::core::BatchHelpers::LoadMessage(filename, dictionary.get());
    dictionary->SerializeToString(last_message());
    return last_message()->size();
  } CATCH_EXCEPTIONS;
}
