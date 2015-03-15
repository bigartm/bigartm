// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/generation.h"

#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"  // generators

#include "glog/logging.h"

#include "artm/core/common.h"
#include "artm/core/helpers.h"
#include "artm/core/exceptions.h"

namespace artm {
namespace core {

DiskGeneration::DiskGeneration(const std::string& disk_path)
    : disk_path_(disk_path), generation_() {
  std::vector<BatchManagerTask> batches = BatchHelpers::ListAllBatches(disk_path);
  for (int i = 0; i < batches.size(); ++i) {
    generation_.push_back(batches[i]);
  }
}

std::vector<BatchManagerTask> DiskGeneration::batch_uuids() const {
  return generation_;
}

}  // namespace core
}  // namespace artm

