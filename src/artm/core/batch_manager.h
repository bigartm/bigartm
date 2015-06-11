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
#include "artm/core/processor_input.h"

namespace artm {
namespace core {

template<typename T> class ThreadSafeHolder;
class InstanceSchema;

class BatchManager : boost::noncopyable, public Notifiable {
 public:
  explicit BatchManager();
  virtual ~BatchManager() {}

  void Add(const boost::uuids::uuid& task_id, const std::string& file_path,
           const ModelName& model_name);

  // Remove all pending batches related to the model.
  void DisposeModel(const ModelName& model_name);

  // Checks if all added tasks were processed (and marked as "Done").
  bool IsEverythingProcessed() const;

  virtual void Callback(const boost::uuids::uuid& task_id, const ModelName& model_name);

 private:
  mutable boost::mutex lock_;

  std::map<ModelName, std::shared_ptr<std::map<boost::uuids::uuid, std::string>>> in_progress_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_BATCH_MANAGER_H_
