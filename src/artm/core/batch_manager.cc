// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/batch_manager.h"

#include "boost/uuid/string_generator.hpp"

#include "artm/core/instance_schema.h"
#include "artm/core/thread_safe_holder.h"

namespace artm {
namespace core {

BatchManager::BatchManager(ThreadSafeHolder<InstanceSchema>* schema)
    : lock_(), tasks_(), in_progress_(), schema_(schema) {}

void BatchManager::Add(const boost::uuids::uuid& id) {
  boost::lock_guard<boost::mutex> guard(lock_);
  tasks_.push_back(id);
}

void BatchManager::DisposeModel(const ModelName& model_name) {
  boost::lock_guard<boost::mutex> guard(lock_);
  in_progress_.erase(model_name);
}

boost::uuids::uuid BatchManager::Next() {
  boost::lock_guard<boost::mutex> guard(lock_);
  for (auto iter = tasks_.begin(); iter != tasks_.end(); ++iter) {
    bool task_is_in_progress = false;

    // Check if any model still processes the batch.
    for (auto model_iter = in_progress_.begin();
         model_iter != in_progress_.end();
         ++model_iter) {
      if (model_iter->second->find(*iter) != model_iter->second->end()) {
        task_is_in_progress = true;
        break;
      }
    }

    if (!task_is_in_progress) {
      boost::uuids::uuid retval = *iter;
      tasks_.erase(iter);
      std::vector<ModelName> models = schema_->get()->GetModelNames();
      for (auto &model_name : models) {
        auto model_iter = in_progress_.find(model_name);
        if (model_iter == in_progress_.end()) {
          in_progress_.insert(std::make_pair(
            model_name, std::make_shared<std::set<boost::uuids::uuid>>()));
          model_iter = in_progress_.find(model_name);
        }

        model_iter->second->insert(retval);
      }

      return retval;
    }
  }

  return boost::uuids::uuid();
}

void BatchManager::Done(const boost::uuids::uuid& id, const ModelName& model_name) {
  boost::lock_guard<boost::mutex> guard(lock_);
  if (model_name == ModelName()) {
    for (auto model_iter = in_progress_.begin();
         model_iter != in_progress_.end();
         ++model_iter) {
      model_iter->second->erase(id);
    }
  } else {
    auto model_iter = in_progress_.find(model_name);
    if (model_iter != in_progress_.end()) {
      model_iter->second->erase(id);
    }
  }
}

bool BatchManager::IsEverythingProcessed() const {
  boost::lock_guard<boost::mutex> guard(lock_);
  if (!tasks_.empty()) {
    return false;
  }

  for (auto model_iter = in_progress_.begin();
       model_iter != in_progress_.end();
       ++model_iter) {
    if (!model_iter->second->empty()) {
      return false;
    }
  }

  return true;
}

void BatchManager::Callback(std::shared_ptr<const ModelIncrement> model_increment) {
  for (int batch_index = 0; batch_index < model_increment->batch_uuid_size(); ++batch_index) {
    std::string uuid_str = model_increment->batch_uuid(batch_index);
    boost::uuids::uuid uuid(boost::uuids::string_generator()(uuid_str.c_str()));

    ModelName model_name = model_increment->model_name();
    Done(uuid, model_name);
  }
}

}  // namespace core
}  // namespace artm
