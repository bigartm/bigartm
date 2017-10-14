// Copyright 2017, Additive Regularization of Topic Models.

#include "artm/core/batch_manager.h"

#include "artm/core/thread_safe_holder.h"

namespace artm {
namespace core {

BatchManager::BatchManager() : lock_(), in_progress_() { }

void BatchManager::Add(const boost::uuids::uuid& task_id) {
  boost::lock_guard<boost::mutex> guard(lock_);
  in_progress_.insert(task_id);
}

bool BatchManager::IsEverythingProcessed() const {
  boost::lock_guard<boost::mutex> guard(lock_);
  return in_progress_.empty();
}

void BatchManager::Callback(const boost::uuids::uuid& task_id) {
  boost::lock_guard<boost::mutex> guard(lock_);
  in_progress_.erase(task_id);
}

}  // namespace core
}  // namespace artm
