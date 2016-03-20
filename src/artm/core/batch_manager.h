// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_BATCH_MANAGER_H_
#define SRC_ARTM_CORE_BATCH_MANAGER_H_

#include <set>
#include <string>

#include "boost/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"
#include "boost/uuid/uuid.hpp"

#include "artm/core/common.h"
#include "artm/core/processor_input.h"

namespace artm {
namespace core {

// BatchManager class keeps track of ongoing tasks.
// Each task is typically associated with processing a specific batch
// the UUID 'task_id' then the same as 'batch.id' field.
class BatchManager : boost::noncopyable {
 public:
  BatchManager();

  // Adds task for execution
  void Add(const boost::uuids::uuid& task_id);

  // Checks if all added tasks were processed
  bool IsEverythingProcessed() const;

  // Marks task as completed
  void Callback(const boost::uuids::uuid& task_id);

 private:
  mutable boost::mutex lock_;
  std::set<boost::uuids::uuid> in_progress_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_BATCH_MANAGER_H_
