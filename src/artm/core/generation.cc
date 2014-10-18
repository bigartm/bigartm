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

boost::uuids::uuid DiskGeneration::AddBatch(const std::shared_ptr<Batch>& batch) {
  std::string message = "ArtmAddBatch() is not allowed with current configuration. ";
  message += "Please, set the configuration parameter MasterComponentConfig.disk_path ";
  message += "to an empty string in order to enable ArtmAddBatch() operation. ";
  message += "Use ArtmSaveBatch() operation to save batches to disk.";
  BOOST_THROW_EXCEPTION(InvalidOperation(message));
}

void DiskGeneration::RemoveBatch(const boost::uuids::uuid& uuid) {
  LOG(ERROR) << "Remove batch is not supported in disk generation.";
}

std::vector<boost::uuids::uuid> DiskGeneration::batch_uuids() const {
  return generation_;
}

std::shared_ptr<Batch> DiskGeneration::batch(const boost::uuids::uuid& uuid) const {
  return BatchHelpers::LoadBatch(uuid, disk_path_);
}

std::shared_ptr<Batch> MemoryGeneration::batch(const boost::uuids::uuid& uuid) const {
  return generation_.get(uuid);
}

std::vector<boost::uuids::uuid> MemoryGeneration::batch_uuids() const {
  return generation_.keys();
}

boost::uuids::uuid MemoryGeneration::AddBatch(const std::shared_ptr<Batch>& batch) {
  boost::uuids::uuid retval = boost::uuids::random_generator()();
  generation_.set(retval, batch);
  return retval;
}

void MemoryGeneration::RemoveBatch(const boost::uuids::uuid& uuid) {
  generation_.erase(uuid);
}

int MemoryGeneration::GetTotalItemsCount() const {
  auto keys = generation_.keys();
  int retval = 0;
  for (auto& key : keys) {
    auto value = generation_.get(key);
    if (value != nullptr) retval += value->item_size();
  }

  return retval;
}

}  // namespace core
}  // namespace artm

