// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_ZMQ_CONTEXT_H_
#define SRC_ARTM_CORE_ZMQ_CONTEXT_H_

#include <memory>
#include "boost/utility.hpp"
#include "glog/logging.h"
#include "zmq.hpp"

namespace artm {
namespace core {

class ZmqContext : boost::noncopyable {
 public:
  static ZmqContext& singleton() {
    static ZmqContext object;
    return object;
  }

  ~ZmqContext() {
    zmq_context_.release();  // workaround for https://github.com/sashafrey/topicmod/issues/54
    LOG(INFO) << "ZeroMQ context destroyed";
  }

  zmq::context_t* get() { return zmq_context_.get(); }

 private:
  // Singleton (make constructor private)
  ZmqContext() : zmq_context_(new zmq::context_t(1)) {
    LOG(INFO) << "ZeroMQ context created";
  }

  std::unique_ptr<zmq::context_t> zmq_context_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_ZMQ_CONTEXT_H_
