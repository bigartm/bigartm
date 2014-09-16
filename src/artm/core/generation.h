// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_GENERATION_H_
#define SRC_ARTM_CORE_GENERATION_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/uuid/uuid.hpp"

#include "artm/messages.pb.h"

namespace artm {
namespace core {

class Generation {
 public:
  virtual std::vector<boost::uuids::uuid> batch_uuids() const = 0;
  virtual std::shared_ptr<const Batch> batch(const boost::uuids::uuid& uuid) const = 0;
  virtual bool empty() const = 0;
  virtual int GetTotalItemsCount() const = 0;
  virtual std::shared_ptr<Generation> Clone() const = 0;
  virtual void AddBatch(const std::shared_ptr<const Batch>& batch) = 0;
};

class DiskGeneration : public Generation {
 public:
  explicit DiskGeneration(const std::string& disk_path);

  virtual std::vector<boost::uuids::uuid> batch_uuids() const;
  virtual std::shared_ptr<const Batch> batch(const boost::uuids::uuid& uuid) const;

  virtual std::shared_ptr<Generation> Clone() const;
  virtual void AddBatch(const std::shared_ptr<const Batch>& batch);
  virtual int GetTotalItemsCount() const { return 0; }
  virtual bool empty() const { return generation_.empty(); }

 private:
  std::string disk_path_;
  std::vector<boost::uuids::uuid> generation_;
};

class MemoryGeneration : public Generation {
 public:
  virtual std::vector<boost::uuids::uuid> batch_uuids() const;
  virtual std::shared_ptr<const Batch> batch(const boost::uuids::uuid& uuid) const;

  virtual std::shared_ptr<Generation> Clone() const;
  virtual void AddBatch(const std::shared_ptr<const Batch>& batch);

  virtual bool empty() const { return generation_.empty(); }
  virtual int GetTotalItemsCount() const;

 private:
  std::map<boost::uuids::uuid, std::shared_ptr<const Batch> > generation_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_GENERATION_H_
