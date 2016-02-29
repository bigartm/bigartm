// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/batch_manager.h"

#include "artm/core/thread_safe_holder.h"

namespace artm {
namespace core {

BatchManager::BatchManager() : lock_(), in_progress_() {}

void BatchManager::DisposeModel(const ModelName& model_name) {
  boost::lock_guard<boost::mutex> guard(lock_);
  in_progress_.erase(model_name);
}

void BatchManager::Add(const boost::uuids::uuid& task_id, const std::string& file_path,
                       const ModelName& model_name) {
  boost::lock_guard<boost::mutex> guard(lock_);

  auto model_iter = in_progress_.find(model_name);
  if (model_iter == in_progress_.end()) {
    in_progress_.insert(std::make_pair(
        model_name, std::make_shared<std::map<boost::uuids::uuid, std::string>>()));
    model_iter = in_progress_.find(model_name);
  }

  model_iter->second->insert(std::make_pair(task_id, file_path));
}

bool BatchManager::IsEverythingProcessed() const {
  boost::lock_guard<boost::mutex> guard(lock_);

  for (auto model_iter = in_progress_.begin();
       model_iter != in_progress_.end();
       ++model_iter) {
    if (!model_iter->second->empty()) {
      return false;
    }
  }

  return true;
}

void BatchManager::Callback(const boost::uuids::uuid& task_id, const ModelName& model_name) {
  boost::lock_guard<boost::mutex> guard(lock_);
  if (model_name == ModelName()) {
    for (auto model_iter = in_progress_.begin();
         model_iter != in_progress_.end();
         ++model_iter) {
      model_iter->second->erase(task_id);
    }
  } else {
    auto model_iter = in_progress_.find(model_name);
    if (model_iter != in_progress_.end()) {
      model_iter->second->erase(task_id);
    }
  }
}

}  // namespace core
}  // namespace artm
