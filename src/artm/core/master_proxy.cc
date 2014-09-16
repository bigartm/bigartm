// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/master_proxy.h"

#include "rpcz/application.hpp"

#include "artm/core/zmq_context.h"


namespace artm {
namespace core {

MasterProxy::MasterProxy(int id, const MasterProxyConfig& config)
    : id_(id),
      communication_timeout_(config.communication_timeout()),
      polling_frequency_(config.polling_frequency()) {
  rpcz::application::options options(3);
  options.zeromq_context = ZmqContext::singleton().get();
  application_.reset(new rpcz::application(options));

  node_controller_service_proxy_.reset(
    new artm::core::NodeControllerService_Stub(
      application_->create_rpc_channel(config.node_connect_endpoint()), true));

  make_rpcz_call([&]() {
    Void response;

    // Reset the state of the remote node controller
    node_controller_service_proxy_->DisposeInstance(Void(), &response, communication_timeout_);
  }, "(within CreateOrReconfigureMasterComponent) DisposeInstance");

  make_rpcz_call([&]() {
    Void response;

    // Reset the state of the remote node controller
    node_controller_service_proxy_->DisposeMasterComponent(Void(), &response, communication_timeout_);
  }, "(within CreateOrReconfigureMasterComponent) DisposeMasterComponent");

  make_rpcz_call([&]() {
    Void response;

    // Create master component on the remote node controller
    node_controller_service_proxy_->CreateOrReconfigureMasterComponent(
      config.config(), &response, communication_timeout_);
  }, "CreateOrReconfigureMasterComponent");
}
MasterProxy::~MasterProxy() {
  make_rpcz_call_no_throw([&]() {
    Void response;
    node_controller_service_proxy_->DisposeMasterComponent(
      Void(), &response, communication_timeout_);
  }, "DisposeMasterComponent");
}

int MasterProxy::id() const { return id_; }

void MasterProxy::Reconfigure(const MasterComponentConfig& config) {
  make_rpcz_call([&]() {
    Void response;
    node_controller_service_proxy_->CreateOrReconfigureMasterComponent(
      config, &response, communication_timeout_);
  }, "CreateOrReconfigureMasterComponent");
}

void MasterProxy::CreateOrReconfigureModel(const ModelConfig& config) {
  make_rpcz_call([&]() {
    CreateOrReconfigureModelArgs request;
    request.set_model_name(config.name());
    request.mutable_config()->CopyFrom(config);
    Void response;
    node_controller_service_proxy_->CreateOrReconfigureModel(
      request, &response, communication_timeout_);
  }, "CreateOrReconfigureModel");
}

void MasterProxy::DisposeModel(ModelName model_name) {
  make_rpcz_call_no_throw([&]() {
    DisposeModelArgs args;
    args.set_model_name(model_name);
    Void response;
    node_controller_service_proxy_->DisposeModel(args, &response, communication_timeout_);
  }, "DisposeModel");
}

void MasterProxy::CreateOrReconfigureRegularizer(const RegularizerConfig& config) {
  make_rpcz_call([&]() {
    CreateOrReconfigureRegularizerArgs request;
    request.set_regularizer_name(config.name());
    request.mutable_config()->CopyFrom(config);
    Void response;
    node_controller_service_proxy_->CreateOrReconfigureRegularizer(
      request, &response, communication_timeout_);
  }, "CreateOrReconfigureRegularizer");
}

void MasterProxy::DisposeRegularizer(const std::string& name) {
  make_rpcz_call_no_throw([&]() {
    DisposeRegularizerArgs args;
    args.set_regularizer_name(name);
    Void response;
    node_controller_service_proxy_->DisposeRegularizer(args, &response, communication_timeout_);
  }, "DisposeRegularizer");
}

void MasterProxy::CreateOrReconfigureDictionary(const DictionaryConfig& config) {
  make_rpcz_call([&]() {
    CreateOrReconfigureDictionaryArgs request;
    request.mutable_dictionary()->CopyFrom(config);
    Void response;
    node_controller_service_proxy_->CreateOrReconfigureDictionary(
      request, &response, communication_timeout_);
  }, "CreateOrReconfigureDictionary");
}

void MasterProxy::DisposeDictionary(const std::string& name) {
  make_rpcz_call_no_throw([&]() {
    DisposeDictionaryArgs args;
    args.set_dictionary_name(name);
    Void response;
    node_controller_service_proxy_->DisposeDictionary(args, &response, communication_timeout_);
  }, "DisposeDictionary");
}

void MasterProxy::OverwriteTopicModel(const ::artm::TopicModel& topic_model) {
  make_rpcz_call([&]() {
    Void response;
    node_controller_service_proxy_->OverwriteTopicModel(
      topic_model, &response, communication_timeout_);
  }, "CreateOrReconfigureDictionary");
}

bool MasterProxy::RequestTopicModel(ModelName model_name, ::artm::TopicModel* topic_model) {
  make_rpcz_call([&]() {
    String request;
    request.set_value(model_name);
    node_controller_service_proxy_->RequestTopicModel(
      request, topic_model, communication_timeout_);
  }, "RequestTopicModel");

  return true;
}

void MasterProxy::RequestRegularizerState(RegularizerName regularizer_name,
                                          ::artm::RegularizerInternalState* regularizer_state) {
  make_rpcz_call([&]() {
    String request;
    request.set_value(regularizer_name);
    node_controller_service_proxy_->RequestRegularizerState(
      request, regularizer_state, communication_timeout_);
  }, "RequestRegularizerState");
}

bool MasterProxy::RequestThetaMatrix(ModelName model_name, ::artm::ThetaMatrix* theta_matrix) {
  make_rpcz_call([&]() {
    String request;
    request.set_value(model_name);
    node_controller_service_proxy_->RequestThetaMatrix(
      request, theta_matrix, communication_timeout_);
  }, "RequestThetaMatrix");

  return true;
}

bool MasterProxy::RequestScore(const ModelName& model_name, const ScoreName& score_name,
                               ::artm::ScoreData* score_data) {
  make_rpcz_call([&]() {
    RequestScoreArgs request;
    request.set_model_name(model_name);
    request.set_score_name(score_name);
    node_controller_service_proxy_->RequestScore(request, score_data, communication_timeout_);
  }, "RequestScore");

  return true;
}

void MasterProxy::AddBatch(const Batch& batch) {
  make_rpcz_call([&]() {
    Void response;
    node_controller_service_proxy_->AddBatch(
      batch, &response, communication_timeout_);
  }, "AddBatch");
}

void MasterProxy::InvokeIteration(int iterations_count) {
  make_rpcz_call([&]() {
    Void response;
    node_controller_service_proxy_->InvokeIteration(
      Void(), &response, communication_timeout_);
  }, "InvokeIteration");
}

bool MasterProxy::WaitIdle(int timeout) {
  Int response;
  auto time_start = boost::posix_time::microsec_clock::local_time();
  for (;;) {
    make_rpcz_call([&]() {
      node_controller_service_proxy_->WaitIdle(Void(), &response, communication_timeout_);
    }, "WaitIdle");
    if (response.value() == ARTM_STILL_WORKING) {
      boost::this_thread::sleep(boost::posix_time::milliseconds(polling_frequency_));
      auto time_end = boost::posix_time::microsec_clock::local_time();

      if (timeout >= 0) {
        if ((time_end - time_start).total_milliseconds() >= timeout) return false;
      }
    } else {  // return value is ARTM_SUCCESS
      return true;
    }
  }
}

void MasterProxy::InvokePhiRegularizers() {
  make_rpcz_call([&]() {
    Void response;
    node_controller_service_proxy_->InvokePhiRegularizers(
      Void(), &response, communication_timeout_);
  }, "InvokePhiRegularizers");
}

}  // namespace core
}  // namespace artm

