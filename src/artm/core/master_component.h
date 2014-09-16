// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_MASTER_COMPONENT_H_
#define SRC_ARTM_CORE_MASTER_COMPONENT_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"

#include "rpcz/application.hpp"
#include "rpcz/rpc.hpp"
#include "rpcz/server.hpp"
#include "rpcz/service.hpp"

#include "artm/messages.pb.h"

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"
#include "artm/core/master_interface.h"
#include "artm/core/master_component_service_impl.h"
#include "artm/core/template_manager.h"
#include "artm/core/thread_safe_holder.h"

namespace zmq {
class context_t;
}  // namespace zmq

namespace artm {

class RegularizerInterface;

namespace core {

class NetworkClientCollection;
class Instance;
class TopicModel;
class Score;

class MasterComponent : boost::noncopyable, public MasterInterface {
 public:
  ~MasterComponent();

  virtual int id() const;
  bool isInLocalModusOperandi() const;
  bool isInNetworkModusOperandi() const;

  MasterComponentServiceImpl* impl() { return master_component_service_impl_.get(); }

  // Retrieves topic model.
  // Returns true if succeeded, and false if model_name hasn't been found.
  virtual bool RequestTopicModel(ModelName model_name, ::artm::TopicModel* topic_model);
  virtual void RequestRegularizerState(RegularizerName regularizer_name,
                                       ::artm::RegularizerInternalState* regularizer_state);
  virtual bool RequestThetaMatrix(ModelName model_name, ::artm::ThetaMatrix* theta_matrix);
  virtual bool RequestScore(const ModelName& model_name, const ScoreName& score_name,
                            ScoreData* score_data);

  // Reconfigures topic model if already exists, otherwise creates a new model.
  virtual void CreateOrReconfigureModel(const ModelConfig& config);
  virtual void OverwriteTopicModel(const ::artm::TopicModel& topic_model);

  virtual void DisposeModel(ModelName model_name);
  virtual void Reconfigure(const MasterComponentConfig& config);

  virtual void CreateOrReconfigureRegularizer(const RegularizerConfig& config);
  virtual void DisposeRegularizer(const std::string& name);
  virtual void InvokePhiRegularizers();

  virtual void CreateOrReconfigureDictionary(const DictionaryConfig& config);
  virtual void DisposeDictionary(const std::string& name);

  // Returns false if BigARTM is still processing the collection, otherwise true.
  virtual bool WaitIdle(int timeout = -1);
  virtual void InvokeIteration(int iterations_count);
  virtual void AddBatch(const Batch& batch);

  // Throws InvalidOperation exception if new config is invalid.
  void ValidateConfig(const MasterComponentConfig& config);

 private:
  class ServiceEndpoint : boost::noncopyable {
   public:
    ServiceEndpoint(const std::string& endpoint, MasterComponentServiceImpl* impl);
    ~ServiceEndpoint();
    std::string endpoint() const { return endpoint_; }

   private:
    std::string endpoint_;
    std::unique_ptr<rpcz::application> application_;
    MasterComponentServiceImpl* impl_;

    // Keep all threads at the end of class members
    // (because the order of class members defines initialization order;
    // everything else should be initialized before creating threads).
    boost::thread thread_;

    void ThreadFunction();
  };

  friend class TemplateManager<MasterInterface>;

  // All master components must be created via TemplateManager.
  MasterComponent(int id, const MasterComponentConfig& config);

  bool is_configured_;

  int master_id_;
  ThreadSafeHolder<MasterComponentConfig> config_;

  // Endpoint for clients to talk with master component
  std::shared_ptr<MasterComponentServiceImpl> master_component_service_impl_;
  std::shared_ptr<ServiceEndpoint> service_endpoint_;

  std::shared_ptr<Instance> instance_;
  std::shared_ptr<NetworkClientCollection> network_client_interface_;
};

class NetworkClientCollection {
 public:
  NetworkClientCollection() : lock_(), application_(nullptr), clients_() {}
  ~NetworkClientCollection();

  void CreateOrReconfigureModel(const ModelConfig& config);
  void DisposeModel(ModelName model_name);

  void CreateOrReconfigureRegularizer(const RegularizerConfig& config);
  void DisposeRegularizer(const std::string& name);

  void CreateOrReconfigureDictionary(const DictionaryConfig& config);
  void DisposeDictionary(const std::string& name);

  void Reconfigure(const MasterComponentConfig& config);

  bool ConnectClient(std::string endpoint);
  bool DisconnectClient(std::string endpoint);

  void ForcePullTopicModel();
  void ForcePushTopicModelIncrement();

  std::vector<std::string> endpoints() const { return clients_.keys(); }
  void for_each_client(std::function<void(artm::core::NodeControllerService_Stub&)> f);
  void for_each_endpoint(std::function<void(std::string)> f);

 private:
  mutable boost::mutex lock_;
  std::unique_ptr<rpcz::application> application_;
  ThreadSafeCollectionHolder<std::string, artm::core::NodeControllerService_Stub> clients_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_MASTER_COMPONENT_H_
