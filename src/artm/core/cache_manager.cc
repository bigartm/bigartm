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

CacheManager::CacheManager(const ThreadSafeHolder<InstanceSchema>& schema)
    : schema_(schema), cache_() {}

CacheManager::~CacheManager() {
  auto keys = cache_.keys();
  for (auto &key : keys) {
    auto cache_entry = cache_.get(key);
    if (cache_entry != nullptr && cache_entry->has_filename())
      try { fs::remove(fs::path(cache_entry->filename())); } catch(...) {}
  }
}

void CacheManager::DisposeModel(ModelName model_name) {
  auto keys = cache_.keys();
  for (auto &key : keys) {
    auto cache_entry = cache_.get(key);
    if (cache_entry == nullptr) {
      continue;
    }

    if (cache_entry->model_name() == model_name) {
      if (cache_entry->has_filename())
        try { fs::remove(fs::path(cache_entry->filename())); } catch(...) {}
      cache_.erase(key);
    }
  }
}

bool CacheManager::RequestThetaMatrix(const GetThetaMatrixArgs& get_theta_args,
                                      ::artm::ThetaMatrix* theta_matrix) const {
  std::string model_name = get_theta_args.model_name();
  std::vector<CacheKey> keys = cache_.keys();

  for (auto &key : keys) {
    if (key.second != model_name)
      continue;

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

    if (get_theta_args.clean_cache()) {
      cache_.erase(key);
    }
  }

  return true;
}

std::shared_ptr<DataLoaderCacheEntry> CacheManager::FindCacheEntry(
    const boost::uuids::uuid& batch_uuid, const ModelName& model_name) const {
  CacheKey key(batch_uuid, model_name);
  std::shared_ptr<DataLoaderCacheEntry> retval = cache_.get(key);
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
  ModelName model_name = cache_entry->model_name();
  CacheKey cache_key(uuid, model_name);
  std::shared_ptr<DataLoaderCacheEntry> old_entry = cache_.get(cache_key);
  cache_.set(cache_key, cache_entry);
  if (old_entry != nullptr && old_entry->has_filename())
    try { fs::remove(fs::path(old_entry->filename())); } catch(...) {}
}

}  // namespace core
}  // namespace artm
