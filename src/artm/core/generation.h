// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_GENERATION_H_
#define SRC_ARTM_CORE_GENERATION_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/uuid/uuid.hpp"

#include "artm/messages.pb.h"
#include "artm/core/thread_safe_holder.h"

namespace artm {
namespace core {

class DiskGeneration {
 public:
  explicit DiskGeneration(const std::string& disk_path);

  std::vector<BatchManagerTask> batch_uuids() const;
  static std::shared_ptr<Batch> batch(const BatchManagerTask& task);

  boost::uuids::uuid AddBatch(const std::shared_ptr<Batch>& batch);
  void RemoveBatch(const boost::uuids::uuid& uuid);
  bool empty() const { return generation_.empty(); }

 private:
  std::string disk_path_;
  std::vector<BatchManagerTask> generation_;  // created one in constructor and then does not change.
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_GENERATION_H_
