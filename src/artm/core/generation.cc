// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/generation.h"

#include "boost/uuid/uuid_generators.hpp"  // generators

#include "glog/logging.h"

#include "artm/core/common.h"
#include "artm/core/helpers.h"
#include "artm/core/exceptions.h"

namespace artm {
namespace core {

DiskGeneration::DiskGeneration(const std::string& disk_path)
    : disk_path_(disk_path), generation_() {
  std::vector<boost::uuids::uuid> batch_uuids = BatchHelpers::ListAllBatches(disk_path);
  generation_.swap(batch_uuids);
}

void DiskGeneration::AddBatch(const std::shared_ptr<const Batch>& batch) {
  std::string message = "ArtmAddBatch() is not allowed with current configuration. ";
  message += "Please, set the configuration parameter MasterComponentConfig.disk_path ";
  message += "to an empty string in order to enable ArtmAddBatch() operation. ";
  message += "Use ArtmSaveBatch() operation to save batches to disk.";
  BOOST_THROW_EXCEPTION(InvalidOperation(message));
}

std::vector<boost::uuids::uuid> DiskGeneration::batch_uuids() const {
  return generation_;
}

std::shared_ptr<const Batch> DiskGeneration::batch(const boost::uuids::uuid& uuid) const {
  return BatchHelpers::LoadBatch(uuid, disk_path_);
}

std::shared_ptr<Generation> DiskGeneration::Clone() const {
  return std::shared_ptr<Generation>(new DiskGeneration(*this));
}

std::shared_ptr<const Batch> MemoryGeneration::batch(const boost::uuids::uuid& uuid) const {
  auto retval = generation_.find(uuid);
  if (retval != generation_.end()) {
    return retval->second;
  }

  return nullptr;
}

std::vector<boost::uuids::uuid> MemoryGeneration::batch_uuids() const {
  std::vector<boost::uuids::uuid> retval;
  for (auto iter = generation_.begin(); iter != generation_.end(); ++iter) {
    retval.push_back(iter->first);
  }

  return std::move(retval);
}

void MemoryGeneration::AddBatch(const std::shared_ptr<const Batch>& batch) {
  generation_.insert(std::make_pair(boost::uuids::random_generator()(), batch));
}

int MemoryGeneration::GetTotalItemsCount() const {
  int retval = 0;
  for (auto iter = generation_.begin(); iter != generation_.end(); ++iter) {
    if ((*iter).second != nullptr) {
      retval += (*iter).second->item_size();
    }
  }

  return retval;
}

std::shared_ptr<Generation> MemoryGeneration::Clone() const {
  return std::shared_ptr<Generation>(new MemoryGeneration(*this));
}

}  // namespace core
}  // namespace artm

