// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/cache_manager.h"

#include "boost/filesystem.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

#include "artm/core/helpers.h"

namespace fs = boost::filesystem;

namespace artm {
namespace core {

CacheManager::CacheManager() : cache_() {}

CacheManager::~CacheManager() {
  auto keys = cache_.keys();
  for (auto &key : keys) {
    auto cache_entry = cache_.get(key);
    if (cache_entry != nullptr && cache_entry->has_filename()) {
      try { fs::remove(fs::path(cache_entry->filename())); } catch(...) {}
    }
  }
}

void CacheManager::RequestMasterComponentInfo(MasterComponentInfo* master_info) const {
  for (auto& key : cache_.keys()) {
    std::shared_ptr<DataLoaderCacheEntry> entry = cache_.get(key);
    if (entry == nullptr)
      continue;

    MasterComponentInfo::CacheEntryInfo* info = master_info->add_cache_entry();
    info->set_key(boost::lexical_cast<std::string>(key));
    info->set_byte_size(entry->ByteSize());
  }
}

void CacheManager::Clear() {
  auto keys = cache_.keys();
  for (auto &key : keys) {
    auto cache_entry = cache_.get(key);
    if (cache_entry != nullptr && cache_entry->has_filename()) {
      try { fs::remove(fs::path(cache_entry->filename())); } catch(...) {}
    }
  }

  cache_.clear();
}

void CacheManager::RequestThetaMatrix(const GetThetaMatrixArgs& get_theta_args,
                                      ::artm::ThetaMatrix* theta_matrix) const {
  auto keys = cache_.keys();
  for (auto &key : keys) {
    std::shared_ptr<DataLoaderCacheEntry> cache = cache_.get(key);
    if (cache == nullptr)
      continue;

    if (cache->has_filename()) {
      DataLoaderCacheEntry cache_reloaded;
      BatchHelpers::LoadMessage(cache->filename(), &cache_reloaded);
      BatchHelpers::PopulateThetaMatrixFromCacheEntry(cache_reloaded, get_theta_args, theta_matrix);
    } else {
      BatchHelpers::PopulateThetaMatrixFromCacheEntry(*cache, get_theta_args, theta_matrix);
    }
  }
}

std::shared_ptr<DataLoaderCacheEntry> CacheManager::FindCacheEntry(
    const boost::uuids::uuid& batch_uuid) const {
  std::shared_ptr<DataLoaderCacheEntry> retval = cache_.get(batch_uuid);
  if (retval == nullptr || !retval->has_filename())
    return retval;

  try {
    std::shared_ptr<DataLoaderCacheEntry> copy(std::make_shared<DataLoaderCacheEntry>());
    copy->CopyFrom(*retval);
    BatchHelpers::LoadMessage(retval->filename(), copy.get());
    // copy->clear_filename();
    return copy;
  } catch(...) {
    LOG(ERROR) << "Unable to reload cache for " << retval->filename();
  }
}

void CacheManager::UpdateCacheEntry(std::shared_ptr<DataLoaderCacheEntry> cache_entry) const {
  std::string uuid_str = cache_entry->batch_uuid();
  boost::uuids::uuid uuid(boost::uuids::string_generator()(uuid_str.c_str()));
  std::shared_ptr<DataLoaderCacheEntry> old_entry = cache_.get(uuid);
  cache_.set(uuid, cache_entry);
  if (old_entry != nullptr && old_entry->has_filename()) {
    try { fs::remove(fs::path(old_entry->filename())); } catch(...) {}
  }
}

}  // namespace core
}  // namespace artm
