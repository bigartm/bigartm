// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_BATCH_MANAGER_H_
#define SRC_ARTM_CORE_BATCH_MANAGER_H_

#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"
#include "boost/uuid/uuid.hpp"

#include "artm/core/common.h"

namespace artm {
namespace core {

template<typename T> class ThreadSafeHolder;
class InstanceSchema;

// Each batch should be processed by at max one processor at a time.
// Consider a scenario when there is a very slow processor,
// and it keeps processing the batch when DataLoader starts the next iteration.
// In such situation BatchManager will ensure that no other processors will receive the
// batch until slow processor is done.
class BatchManager : boost::noncopyable, public Notifiable {
 public:
  explicit BatchManager(ThreadSafeHolder<InstanceSchema>* schema);
  virtual ~BatchManager() {}

  // Add batch to the task queue.
  // OK to add the same uuid multiple times.
  void Add(const boost::uuids::uuid& id);

  // Remove all pending batches related to the model.
  void DisposeModel(const ModelName& model_name);

  // Select next available batch, and excludes it from task queue.
  // This operation skips all "in progress" batches.
  // The batch return by this operation will stay in "in progress" list
  // until it is marked as processed by Done().
  boost::uuids::uuid Next();

  // Eliminates uuid from "in progress" set.
  void Done(const boost::uuids::uuid& id, const ModelName& model_name);

  // Checks if all added tasks were processed (and marked as "Done").
  bool IsEverythingProcessed() const;

  virtual void Callback(std::shared_ptr<const ModelIncrement> model_increment);

 private:
  mutable boost::mutex lock_;
  std::list<boost::uuids::uuid> tasks_;
  std::map<ModelName, std::shared_ptr<std::set<boost::uuids::uuid>>> in_progress_;
  ThreadSafeHolder<InstanceSchema>* schema_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_BATCH_MANAGER_H_
