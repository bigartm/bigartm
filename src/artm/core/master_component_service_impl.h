// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_MASTER_COMPONENT_SERVICE_IMPL_H_
#define SRC_ARTM_CORE_MASTER_COMPONENT_SERVICE_IMPL_H_

#include "boost/thread/mutex.hpp"

#include "rpcz/service.hpp"

#include "artm/messages.pb.h"
#include "artm/core/common.h"
#include "artm/core/internals.rpcz.h"

namespace zmq {
class context_t;
}  // namespace zmq

namespace rpcz {
class application;
}  // namespace rpcz

namespace artm {
namespace core {

class Instance;

class MasterComponentServiceImpl : public MasterComponentService {
 public:
  explicit MasterComponentServiceImpl(Instance* instance);
  ~MasterComponentServiceImpl() { ; }

  virtual void UpdateModel(const ::artm::core::ModelIncrement& request,
                       ::rpcz::reply< ::artm::core::Void> response);
  virtual void RetrieveModel(const ::artm::core::String& request,
                       ::rpcz::reply< ::artm::TopicModel> response);

  virtual void RequestBatches(const ::artm::core::Int& request,
                       ::rpcz::reply< ::artm::core::BatchIds> response);
  virtual void ReportBatches(const ::artm::core::BatchIds& request,
                       ::rpcz::reply< ::artm::core::Void> response);

 private:
  Instance* instance_;
};

}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_MASTER_COMPONENT_SERVICE_IMPL_H_
