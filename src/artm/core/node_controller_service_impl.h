// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_NODE_CONTROLLER_SERVICE_IMPL_H_
#define SRC_ARTM_CORE_NODE_CONTROLLER_SERVICE_IMPL_H_

#include <memory>

#include "boost/thread/mutex.hpp"

#include "rpcz/service.hpp"

#include "artm/core/common.h"
#include "artm/core/internals.rpcz.h"

namespace artm {
namespace core {

class Instance;
class MasterInterface;

class NodeControllerServiceImpl : public NodeControllerService {
 public:
  NodeControllerServiceImpl();
  ~NodeControllerServiceImpl();
  Instance* instance();

  // The following methods talks to instance_
  virtual void CreateOrReconfigureInstance(const ::artm::MasterComponentConfig& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void DisposeInstance(const ::artm::core::Void& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void ForcePullTopicModel(const ::artm::core::Void& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void ForcePushTopicModelIncrement(const ::artm::core::Void& request,
                       ::rpcz::reply< ::artm::core::Void> response);

  // The following methods talks to master_
  virtual void CreateOrReconfigureMasterComponent(const ::artm::MasterComponentConfig& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void DisposeMasterComponent(const ::artm::core::Void& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void OverwriteTopicModel(const ::artm::TopicModel& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void RequestTopicModel(const ::artm::core::String& request,
                       ::rpcz::reply< ::artm::TopicModel> response);
  virtual void RequestRegularizerState(const ::artm::core::String& request,
                       ::rpcz::reply< ::artm::RegularizerInternalState> response);
  virtual void RequestThetaMatrix(const ::artm::core::String& request,
                       ::rpcz::reply< ::artm::ThetaMatrix> response);
  virtual void RequestScore(const ::artm::core::RequestScoreArgs& request,
                       ::rpcz::reply< ::artm::ScoreData> response);
  virtual void AddBatch(const ::artm::Batch& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void InvokeIteration(const ::artm::core::Void& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void WaitIdle(const ::artm::core::Void& request,
                       ::rpcz::reply< ::artm::core::Int> response);
  virtual void InvokePhiRegularizers(const ::artm::core::Void& request,
                       ::rpcz::reply< ::artm::core::Void> response);

  // The following methods talks to instance_ or master_
  virtual void CreateOrReconfigureModel(const ::artm::core::CreateOrReconfigureModelArgs& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void DisposeModel(const ::artm::core::DisposeModelArgs& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void CreateOrReconfigureRegularizer(const ::artm::core::CreateOrReconfigureRegularizerArgs& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void DisposeRegularizer(const ::artm::core::DisposeRegularizerArgs& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void CreateOrReconfigureDictionary(const ::artm::core::CreateOrReconfigureDictionaryArgs& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void DisposeDictionary(const ::artm::core::DisposeDictionaryArgs& request,
                       ::rpcz::reply< ::artm::core::Void> response);

 private:
  void VerifyCurrentState();

  mutable boost::mutex lock_;

  // Currently node controller supports only one Instance or MasterComponent per node.
  std::shared_ptr<Instance> instance_;
  std::shared_ptr<MasterInterface> master_;
};

}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_NODE_CONTROLLER_SERVICE_IMPL_H_
