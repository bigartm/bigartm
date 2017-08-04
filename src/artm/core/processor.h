// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <atomic>

#include "boost/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"

namespace artm {
namespace core {

class Instance;

// A class that implements ProcessBatch routine from
// 'Parallel Non-blocking Deterministic Algorithm for Online Topic Modeling'.
// Each processor instantiates its own thread that pulls tasks from processor queue, hosted by the Instance.
// Each master components owns its own processors.
// The implementation of the processor class is, perhaps, the most complicated code in BigARTM core.
// If you are looking into Processor then you should consider reading other articles on ARTM theory.
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
