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

// Must be thread-safe (used concurrently in DataLoader).
class Generation {
 public:
  virtual std::vector<BatchManagerTask> batch_uuids() const = 0;
  virtual std::shared_ptr<Batch> batch(const BatchManagerTask& task) const = 0;
  virtual bool empty() const = 0;
  virtual int GetTotalItemsCount() const = 0;
  virtual boost::uuids::uuid AddBatch(const std::shared_ptr<Batch>& batch) = 0;
  virtual void RemoveBatch(const boost::uuids::uuid& uuid) = 0;
};

class DiskGeneration : public Generation {
 public:
  explicit DiskGeneration(const std::string& disk_path);

  virtual std::vector<BatchManagerTask> batch_uuids() const;
  virtual std::shared_ptr<Batch> batch(const BatchManagerTask& task) const;

  virtual boost::uuids::uuid AddBatch(const std::shared_ptr<Batch>& batch);
  virtual void RemoveBatch(const boost::uuids::uuid& uuid);
  virtual int GetTotalItemsCount() const { return 0; }
  virtual bool empty() const { return generation_.empty(); }

 private:
  std::string disk_path_;
  std::vector<BatchManagerTask> generation_;  // created one in constructor and then does not change.
};

class MemoryGeneration : public Generation {
 public:
  virtual std::vector<BatchManagerTask> batch_uuids() const;
  virtual std::shared_ptr<Batch> batch(const BatchManagerTask& task) const;

  virtual boost::uuids::uuid AddBatch(const std::shared_ptr<Batch>& batch);
  virtual void RemoveBatch(const boost::uuids::uuid& uuid);

  virtual bool empty() const { return generation_.empty(); }
  virtual int GetTotalItemsCount() const;

 private:
  ThreadSafeCollectionHolder<boost::uuids::uuid, Batch> generation_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_GENERATION_H_
