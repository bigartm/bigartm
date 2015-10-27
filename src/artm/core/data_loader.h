// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_DATA_LOADER_H_
#define SRC_ARTM_CORE_DATA_LOADER_H_

#include <atomic>
#include <list>
#include <set>
#include <utility>

#include "boost/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"
#include "boost/uuid/uuid.hpp"

#include "artm/messages.pb.h"
#include "artm/core/common.h"
#include "artm/core/thread_safe_holder.h"

namespace artm {
namespace core {

class Instance;

class DataLoader : boost::noncopyable {
 public:
  explicit DataLoader(Instance* instance);
  virtual ~DataLoader();

  bool AddBatch(const AddBatchArgs& args);
  Instance* instance();
  void InvokeIteration(const InvokeIterationArgs& args);

  // Returns false if BigARTM is still processing the collection, otherwise true.
  bool WaitIdle(const WaitIdleArgs& args);

 private:
  Instance* instance_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_DATA_LOADER_H_
