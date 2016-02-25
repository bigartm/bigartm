// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_PROCESSOR_H_
#define SRC_ARTM_CORE_PROCESSOR_H_

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/bind.hpp"
#include "boost/utility.hpp"

#include "artm/messages.pb.h"

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"

#include "artm/utility/blas.h"

namespace artm {
namespace core {

class Instance;

class Processor : boost::noncopyable {
 public:
  explicit Processor(Instance* instance);
  ~Processor();

 private:
  Instance* instance_;

  mutable std::atomic<bool> is_stopping;
  boost::thread thread_;

  void ThreadFunction();
};

}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_PROCESSOR_H_
