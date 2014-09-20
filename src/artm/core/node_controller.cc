// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/node_controller.h"

#include "glog/logging.h"

#include "artm/core/node_controller_service_impl.h"
#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/zmq_context.h"

namespace artm {
namespace core {

NodeController::NodeController(int id, const NodeControllerConfig& config)
    : node_controller_id_(id),
      config_(std::make_shared<NodeControllerConfig>(NodeControllerConfig(config))),
      service_endpoint_(nullptr),
      node_controller_service_impl_() {
  service_endpoint_.reset(new ServiceEndpoint(config.create_endpoint(), impl()));
}

NodeController::~NodeController() {
}

int NodeController::id() const {
  return node_controller_id_;
}

NodeController::ServiceEndpoint::~ServiceEndpoint() {
  application_->terminate();
  thread_.join();
}

NodeController::ServiceEndpoint::ServiceEndpoint(const std::string& endpoint,
                                                 NodeControllerServiceImpl* impl)
    : endpoint_(endpoint), application_(nullptr), impl_(impl), thread_() {
  rpcz::application::options options(3);
  options.zeromq_context = ZmqContext::singleton().get();
  application_.reset(new rpcz::application(options));
  boost::thread t(&NodeController::ServiceEndpoint::ThreadFunction, this);
  thread_.swap(t);
}

void NodeController::ServiceEndpoint::ThreadFunction() {
  try {
    Helpers::SetThreadName(-1, "NodeController");
    LOG(INFO) << "Establishing NodeControllerService on " << endpoint();
    rpcz::server server(*application_);
    server.register_service(impl_);
    server.bind(endpoint());
    application_->run();
    LOG(INFO) << "NodeControllerService on " << endpoint() << " is stopped.";
  } catch(...) {
    LOG(FATAL) << "Fatal exception in NodeController::ServiceEndpoint::ThreadFunction() function";
    return;
  }
}

}  // namespace core
}  // namespace artm
