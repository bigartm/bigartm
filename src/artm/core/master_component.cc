// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/master_component.h"

#include <vector>
#include <set>
#include <sstream>

#include "glog/logging.h"
#include "zmq.hpp"

#include "artm/regularizer_interface.h"
#include "artm/score_calculator_interface.h"

#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/zmq_context.h"
#include "artm/core/data_loader.h"
#include "artm/core/batch_manager.h"
#include "artm/core/instance.h"
#include "artm/core/processor.h"
#include "artm/core/topic_model.h"
#include "artm/core/merger.h"

namespace artm {
namespace core {

MasterComponent::MasterComponent(int id, const MasterComponentConfig& config)
    : is_configured_(false),
      master_id_(id),
      config_(std::make_shared<MasterComponentConfig>(config)),
      master_component_service_impl_(nullptr),
      service_endpoint_(nullptr),
      instance_(nullptr),
      network_client_interface_(nullptr) {
  Reconfigure(config);
}

MasterComponent::~MasterComponent() {
}

int MasterComponent::id() const {
  return master_id_;
}

bool MasterComponent::isInLocalModusOperandi() const {
  return config_.get()->modus_operandi() == MasterComponentConfig_ModusOperandi_Local;
}

bool MasterComponent::isInNetworkModusOperandi() const {
  return config_.get()->modus_operandi() == MasterComponentConfig_ModusOperandi_Network;
}

void MasterComponent::CreateOrReconfigureModel(const ModelConfig& config) {
  if ((config.class_weight_size() != 0 || config.class_id_size() != 0) && !config.use_sparse_bow()) {
    std::stringstream ss;
    ss << "You have configured use_sparse_bow=false. "
       << "Fields ModelConfig.class_id and ModelConfig.class_weight not supported in this mode.";
    BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
  }

  instance_->CreateOrReconfigureModel(config);
  network_client_interface_->CreateOrReconfigureModel(config);
}

void MasterComponent::DisposeModel(ModelName model_name) {
  instance_->DisposeModel(model_name);
  network_client_interface_->DisposeModel(model_name);
}

void MasterComponent::CreateOrReconfigureRegularizer(const RegularizerConfig& config) {
  instance_->CreateOrReconfigureRegularizer(config);
  network_client_interface_->CreateOrReconfigureRegularizer(config);
}

void MasterComponent::DisposeRegularizer(const std::string& name) {
  instance_->DisposeRegularizer(name);
  network_client_interface_->DisposeRegularizer(name);
}

void MasterComponent::CreateOrReconfigureDictionary(const DictionaryConfig& config) {
  instance_->CreateOrReconfigureDictionary(config);
  network_client_interface_->CreateOrReconfigureDictionary(config);
}

void MasterComponent::DisposeDictionary(const std::string& name) {
  instance_->DisposeDictionary(name);
  network_client_interface_->DisposeDictionary(name);
}

void MasterComponent::SynchronizeModel(const SynchronizeModelArgs& args) {
  instance_->merger()->ForceSynchronizeModel(args);

  if (isInNetworkModusOperandi()) {
    network_client_interface_->ForcePullTopicModel();
  }
}

void MasterComponent::InitializeModel(const InitializeModelArgs& args) {
  instance_->merger()->InitializeModel(args);

  if (isInNetworkModusOperandi()) {
    network_client_interface_->ForcePullTopicModel();
  }
}

void MasterComponent::Reconfigure(const MasterComponentConfig& user_config) {
  ValidateConfig(user_config);

  MasterComponentConfig config(user_config);  // make a copy
  if (!config.has_processor_queue_max_size()) {
    // The default setting for processor queue max size is to use the number of processors.
    // This will ensure reasonably good load balancing in network modus operandi.
    config.set_processor_queue_max_size(config.processors_count());
  }

  config_.set(std::make_shared<MasterComponentConfig>(config));

  if (!is_configured_) {
    // First configuration
    bool is_local = (config.modus_operandi() == MasterComponentConfig_ModusOperandi_Local);
    bool is_network = (config.modus_operandi() == MasterComponentConfig_ModusOperandi_Network);
    InstanceType type = (is_local) ? MasterInstanceLocal : MasterInstanceNetwork;
    instance_.reset(new Instance(config, type));

    network_client_interface_.reset(new NetworkClientCollection(config.communication_timeout()));

    if (is_network) {
      master_component_service_impl_.reset(
        new MasterComponentServiceImpl(instance_.get()));

      service_endpoint_.reset(new ServiceEndpoint(
        config.create_endpoint(), master_component_service_impl_.get()));
    }

    is_configured_ = true;
  } else {
    instance_->Reconfigure(config);
  }

  if (isInNetworkModusOperandi()) {
    network_client_interface_->set_communication_timeout(config.communication_timeout());
    std::vector<std::string> vec = network_client_interface_->endpoints();
    std::set<std::string> current_endpoints(vec.begin(), vec.end());
    std::set<std::string> to_be_deleted(vec.begin(), vec.end());

    // Scan through all endpoints in the new config.
    // (1) All endpoints that are NOT present in current_endpoints will be created.
    // (2) All endpoints that are NOT present in config.node_endpoint() list shell be deleted.
    // (3) Endpoints that are present in both lists should stay as they are.
    for (int i = 0; i < config.node_connect_endpoint_size(); ++i) {
      std::string endpoint = config.node_connect_endpoint(i);

      // Endpoint already exists and must not be deleted (3)
      to_be_deleted.erase(endpoint);

      auto iter = current_endpoints.find(endpoint);
      if (iter == current_endpoints.end()) {
        // Endpoint does not exist and must be created (1)
        network_client_interface_->ConnectClient(endpoint);
      }
    }

    for (auto &endpoint : to_be_deleted) {
      // Obsolete endpoints must be deleted (2)
      network_client_interface_->DisconnectClient(endpoint);
    }
  }

  network_client_interface_->Reconfigure(config);
}

bool MasterComponent::RequestTopicModel(const ::artm::GetTopicModelArgs& get_model_args,
                                        ::artm::TopicModel* topic_model) {
  return instance_->merger()->RetrieveExternalTopicModel(get_model_args, topic_model);
}

void MasterComponent::RequestRegularizerState(RegularizerName regularizer_name,
                                              ::artm::RegularizerInternalState* regularizer_state) {
  instance_->merger()->RequestRegularizerState(regularizer_name, regularizer_state);
}

bool MasterComponent::RequestScore(const GetScoreValueArgs& get_score_args,
                                   ScoreData* score_data) {
  if (!get_score_args.has_batch()) {
    return instance_->merger()->RequestScore(get_score_args, score_data);
  }

  if (isInLocalModusOperandi()) {
    if (instance_->processor_size() == 0)
      BOOST_THROW_EXCEPTION(InternalError("No processors exist in the master component"));
    Batch batch(get_score_args.batch());
    BatchHelpers::PopulateClassId(&batch);
    instance_->processor(0)->FindThetaMatrix(batch, GetThetaMatrixArgs(), nullptr, get_score_args, score_data);
    return true;
  }

  if (isInNetworkModusOperandi()) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
      "RequestScore with custom batch is not supported in Network modus operandi"));
  }

  BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
    "MasterComponentConfig.modus_operandi", config_.get()->modus_operandi()));
}

void MasterComponent::OverwriteTopicModel(const ::artm::TopicModel& topic_model) {
  instance_->merger()->OverwriteTopicModel(topic_model);

  if (isInNetworkModusOperandi()) {
    network_client_interface_->ForcePullTopicModel();
  }
}

bool MasterComponent::RequestThetaMatrix(const GetThetaMatrixArgs& get_theta_args,
                                         ::artm::ThetaMatrix* theta_matrix) {
  if (isInLocalModusOperandi()) {
    if (!get_theta_args.has_batch()) {
      return instance_->local_data_loader()->RequestThetaMatrix(get_theta_args, theta_matrix);
    } else {
      if (instance_->processor_size() == 0)
        BOOST_THROW_EXCEPTION(InternalError("No processors exist in the master component"));
      Batch batch(get_theta_args.batch());
      BatchHelpers::PopulateClassId(&batch);
      instance_->processor(0)->FindThetaMatrix(batch, get_theta_args, theta_matrix, GetScoreValueArgs(), nullptr);
      return true;
    }
  }

  if (isInNetworkModusOperandi()) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
      "RequestThetaMatrix is not supported in Network modus operandi"));
  }

  BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
    "MasterComponentConfig.modus_operandi", config_.get()->modus_operandi()));
}

bool MasterComponent::WaitIdle(const WaitIdleArgs& args) {
  int timeout = args.timeout_milliseconds();
  WaitIdleArgs new_args;
  new_args.CopyFrom(args);
  if (isInLocalModusOperandi()) {
    auto time_start = boost::posix_time::microsec_clock::local_time();

    bool retval = instance_->local_data_loader()->WaitIdle(args);
    if (!retval) return false;

    auto time_end = boost::posix_time::microsec_clock::local_time();
    if (timeout != -1) {
      timeout -= (time_end - time_start).total_milliseconds();
      new_args.set_timeout_milliseconds(timeout);
    }
    return instance_->merger()->WaitIdle(new_args);
  }

  if (isInNetworkModusOperandi()) {
    auto time_start = boost::posix_time::microsec_clock::local_time();
    // Wait for all nodes to process all the batches.
    for (;;) {
      if (instance_->batch_manager()->IsEverythingProcessed())
        break;

      boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));
      if (timeout >= 0) {
        auto time_end = boost::posix_time::microsec_clock::local_time();
        if ((time_end - time_start).total_milliseconds() >= timeout) return false;
      }
    }

    {
      CuckooWatch cuckoo("ForcePushTopicModelIncrement");
      // Ask all nodes to push their increments to master
      network_client_interface_->ForcePushTopicModelIncrement();
    }

    {
      CuckooWatch cuckoo("Merge all increments");
      // Wait merger on master to process all model increments and set them as active topic model
      auto time_end = boost::posix_time::microsec_clock::local_time();
      auto local_timeout = timeout - (time_end - time_start).total_milliseconds();
      if (timeout >= 0) {
        if (local_timeout >= 0) {
          new_args.set_timeout_milliseconds(static_cast<int>(local_timeout));
          bool result = instance_->merger()->WaitIdle(new_args);
          if (!result) return false;
        } else {
          return false;
        }
      } else {
        instance_->merger()->WaitIdle(args);
      }
    }

    return true;
  }

  BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
    "MasterComponentConfig.modus_operandi", config_.get()->modus_operandi()));
}

void MasterComponent::InvokeIteration(const InvokeIterationArgs& args) {
  if (args.reset_scores())
    instance_->merger()->ForceResetScores(ModelName());

  if (isInLocalModusOperandi()) {
    instance_->local_data_loader()->InvokeIteration(args);
    return;
  }

  if (isInNetworkModusOperandi()) {
    auto uuids = BatchHelpers::ListAllBatches(config_.get()->disk_path());
    for (int iter = 0; iter < args.iterations_count(); ++iter) {
      for (auto &uuid : uuids) {
        instance_->batch_manager()->Add(uuid);
      }
    }

    return;
  }

  BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
    "MasterComponentConfig.modus_operandi", config_.get()->modus_operandi()));
}

bool MasterComponent::AddBatch(const AddBatchArgs& args) {
  int timeout = args.timeout_milliseconds();
  if (isInLocalModusOperandi()) {
    if (args.reset_scores())
      instance_->merger()->ForceResetScores(ModelName());

    return instance_->local_data_loader()->AddBatch(args);
  }

  if (isInNetworkModusOperandi()) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
      "AddBatch() is not allowed together with MasterComponentConfig.modus_operandi=Network"));
  }

  BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
    "MasterComponentConfig.modus_operandi", config_.get()->modus_operandi()));
}

MasterComponent::ServiceEndpoint::~ServiceEndpoint() {
  application_->terminate();
  thread_.join();
}

MasterComponent::ServiceEndpoint::ServiceEndpoint(
    const std::string& endpoint, MasterComponentServiceImpl* impl)
    : endpoint_(endpoint), application_(nullptr), impl_(impl), thread_() {
  rpcz::application::options options(3);
  options.zeromq_context = ZmqContext::singleton().get();
  application_.reset(new rpcz::application(options));
  boost::thread t(&MasterComponent::ServiceEndpoint::ThreadFunction, this);
  thread_.swap(t);
}

void MasterComponent::ServiceEndpoint::ThreadFunction() {
  try {
    Helpers::SetThreadName(-1, "MasterComponent");
    LOG(INFO) << "Establishing MasterComponentService on " << endpoint();
    rpcz::server server(*application_);
    server.register_service(impl_);
    server.bind(endpoint());
    application_->run();
    LOG(INFO) << "MasterComponentService on " << endpoint() << " is stopped.";
  } catch(...) {
    LOG(FATAL) << "Fatal exception in MasterComponent::ServiceEndpoint::ThreadFunction() function";
    return;
  }
}


void MasterComponent::ValidateConfig(const MasterComponentConfig& config) {
  if (!is_configured_) {
    if ((config.modus_operandi() != MasterComponentConfig_ModusOperandi_Local) &&
        (config.modus_operandi() != MasterComponentConfig_ModusOperandi_Network)) {
      BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
        "MasterComponentConfig.modus_operandi", config_.get()->modus_operandi()));
    }
    if (config.modus_operandi() == MasterComponentConfig_ModusOperandi_Network) {
      if (!config.has_create_endpoint() ||
          !config.has_connect_endpoint()) {
        BOOST_THROW_EXCEPTION(InvalidOperation(
          "Network modus operandi require endpoint to be set."));
      }

      if (!config.has_disk_path()) {
        BOOST_THROW_EXCEPTION(InvalidOperation(
          "Network modus operandi require disk_path to be set."));
      }
    }
  }

  if (is_configured_) {
    std::shared_ptr<MasterComponentConfig> current_config = config_.get();
    if (current_config->modus_operandi() != config.modus_operandi()) {
      BOOST_THROW_EXCEPTION(InvalidOperation("Unable to change modus operandi"));
    }

    if (current_config->create_endpoint() != config.create_endpoint()) {
      BOOST_THROW_EXCEPTION(InvalidOperation("Unable to change master component create endpoint"));
    }

    if (current_config->connect_endpoint() != config.connect_endpoint()) {
      std::string message = "Unable to change master component connect endpoint";
      BOOST_THROW_EXCEPTION(InvalidOperation(message));
    }

    if (current_config->disk_path() != config.disk_path()) {
      std::string message = "Changing disk_path is not supported.";
      BOOST_THROW_EXCEPTION(InvalidOperation(message));
    }
  }
}

void NetworkClientCollection::CreateOrReconfigureModel(const ModelConfig& config) {
  for_each_client([&](NodeControllerService_Stub& client) {
    CreateOrReconfigureModelArgs args;
    args.mutable_config()->CopyFrom(config);
    Void response;
    make_rpcz_call_no_throw([&]() {
      client.CreateOrReconfigureModel(args, &response, communication_timeout());
    }, "NetworkClientCollection::CreateOrReconfigureModel");
  });
}


void NetworkClientCollection::DisposeModel(ModelName model_name) {
  for_each_client([&](NodeControllerService_Stub& client) {
    DisposeModelArgs args;
    args.set_model_name(model_name);
    Void response;
    make_rpcz_call_no_throw([&]() {
      client.DisposeModel(args, &response, communication_timeout());
    }, "NetworkClientCollection::DisposeModel");
  });
}

void NetworkClientCollection::CreateOrReconfigureRegularizer(const RegularizerConfig& config) {
  for_each_client([&](NodeControllerService_Stub& client) {
    CreateOrReconfigureRegularizerArgs args;
    args.mutable_config()->CopyFrom(config);
    Void response;
    make_rpcz_call_no_throw([&]() {
      client.CreateOrReconfigureRegularizer(args, &response, communication_timeout());
    }, "NetworkClientCollection::CreateOrReconfigureRegularizer");
  });
}

void NetworkClientCollection::DisposeRegularizer(const std::string& name) {
  for_each_client([&](NodeControllerService_Stub& client) {
    DisposeRegularizerArgs args;
    args.set_regularizer_name(name);
    Void response;
    make_rpcz_call_no_throw([&]() {
      client.DisposeRegularizer(args, &response, communication_timeout());
    }, "NetworkClientCollection::DisposeRegularizer");
  });
}

void NetworkClientCollection::CreateOrReconfigureDictionary(const DictionaryConfig& config) {
  for_each_client([&](NodeControllerService_Stub& client) {
    CreateOrReconfigureDictionaryArgs args;
    args.mutable_dictionary()->CopyFrom(config);
    Void response;
    make_rpcz_call_no_throw([&]() {
      client.CreateOrReconfigureDictionary(args, &response, communication_timeout());
    }, "NetworkClientCollection::CreateOrReconfigureDictionary");
  });
}

void NetworkClientCollection::DisposeDictionary(const std::string& name) {
  for_each_client([&](NodeControllerService_Stub& client) {
    DisposeDictionaryArgs args;
    args.set_dictionary_name(name);
    Void response;
    make_rpcz_call_no_throw([&]() {
      client.DisposeDictionary(args, &response, communication_timeout());
    }, "NetworkClientCollection::DisposeDictionary");
  });
}

void NetworkClientCollection::Reconfigure(const MasterComponentConfig& config) {
  for_each_client([&](NodeControllerService_Stub& client) {
    Void response;
    make_rpcz_call_no_throw([&]() {
      client.CreateOrReconfigureInstance(config, &response, config.communication_timeout());
    }, "NetworkClientCollection::Reconfigure");
  });
}

NetworkClientCollection::~NetworkClientCollection() {
  for_each_endpoint([&](std::string endpoint) {
    DisconnectClient(endpoint);
  });
}

bool NetworkClientCollection::ConnectClient(std::string endpoint) {
  if (clients_.has_key(endpoint)) {
    LOG(ERROR) << "Unable to connect client " << endpoint << ", client already exists.";
    return false;
  }

  {
    // application_ is created "on demand" (upon first call to ConnectClient).
    // The reason is to avoid any rpcz object when master component is in local modus operandi.
    // The lock_ is needed when two ConnectClient() calls happen from two threads simultaneously.
    boost::lock_guard<boost::mutex> guard(lock_);
    if (application_ == nullptr) {
      rpcz::application::options options(3);
      options.zeromq_context = ZmqContext::singleton().get();
      application_.reset(new rpcz::application(options));
    }
  }

  std::shared_ptr<NodeControllerService_Stub> client(
    new artm::core::NodeControllerService_Stub(
      application_->create_rpc_channel(endpoint), true));


  try {
    // Reset the state of the remote node controller
    Void response;
    make_rpcz_call([&]() { client->DisposeInstance(Void(), &response, communication_timeout()); });
    make_rpcz_call([&]() { client->DisposeMasterComponent(Void(), &response, communication_timeout()); });
  } catch(...) {
    LOG(ERROR) << "Unable to clear the state of the remote node controller.";
    return false;
  }

  clients_.set(endpoint, client);
  return true;
}

bool NetworkClientCollection::DisconnectClient(std::string endpoint) {
  auto client = clients_.get(endpoint);
  if (client != nullptr) {
    Void response;
    make_rpcz_call_no_throw([&]() {
      client->DisposeInstance(Void(), &response, communication_timeout());
    }, "NetworkClientCollection::DisconnectClient");
  } else {
    LOG(ERROR) << "Unable to disconnect client " << endpoint << ", client is not connected.";
    return false;
  }

  clients_.erase(endpoint);
  return true;
}

void NetworkClientCollection::for_each_client(
    std::function<void(artm::core::NodeControllerService_Stub&)> f) {
  std::vector<std::string> client_ids(clients_.keys());
  for (auto &client_id : client_ids) {
    std::shared_ptr<NodeControllerService_Stub> client = clients_.get(client_id);
    if (client != nullptr) {
      f(*client);
    }
  }
}

void NetworkClientCollection::for_each_endpoint(
    std::function<void(std::string)> f) {
  std::vector<std::string> client_ids(clients_.keys());
  for (auto &client_id : client_ids) {
    f(client_id);
  }
}

void NetworkClientCollection::ForcePullTopicModel() {
  for_each_client([&](NodeControllerService_Stub& client) {
    Void response;
    make_rpcz_call_no_throw([&]() {
      client.ForcePullTopicModel(Void(), &response, communication_timeout());
    }, "NetworkClientCollection::ForcePullTopicModel");
  });
}

void NetworkClientCollection::ForcePushTopicModelIncrement() {
  for_each_client([&](NodeControllerService_Stub& client) {
    Void response;
    make_rpcz_call_no_throw([&]() {
      client.ForcePushTopicModelIncrement(Void(), &response, communication_timeout());
    }, "NetworkClientCollection::ForcePushTopicModelIncrement");
  });
}

}  // namespace core
}  // namespace artm
