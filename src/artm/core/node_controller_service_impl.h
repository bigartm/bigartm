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

class NodeControllerServiceImpl : public NodeControllerService {
 public:
  NodeControllerServiceImpl();
  ~NodeControllerServiceImpl();
  Instance* instance();

  virtual void CreateOrReconfigureInstance(const ::artm::MasterComponentConfig& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void DisposeInstance(const ::artm::core::Void& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void ForcePullTopicModel(const ::artm::core::Void& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void ForcePushTopicModelIncrement(const ::artm::core::Void& request,
                       ::rpcz::reply< ::artm::core::Void> response);
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

  // Currently node controller supports only one Instance per node.
  std::shared_ptr<Instance> instance_;
};

}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_NODE_CONTROLLER_SERVICE_IMPL_H_
