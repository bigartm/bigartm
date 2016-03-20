// Copyright 2015, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_CACHE_MANAGER_H_
#define SRC_ARTM_CORE_CACHE_MANAGER_H_

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/utility.hpp"
#include "boost/uuid/uuid.hpp"

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"
#include "artm/core/thread_safe_holder.h"

namespace artm {
namespace core {

// CacheManager class is responsible for caching ThetaMatrix in between calls to different APIs.
// This class is used when the user calls FitOffline / FitOnline / Transfor to store the resulting theta matrix.
// (at least when theta_matrix_type is set to ThetaMatrixType_Cache).
// Later user may retrieve the data from CacheManager via calls to ArtmRequestThetaMatrix.
// The cache is organized as a set of entries, each entry associated with a single batch.
// The key in the cache corresponds to 'batch.id' field.
class CacheManager : boost::noncopyable {
 public:
  CacheManager();
  virtual ~CacheManager();

  void RequestMasterComponentInfo(MasterComponentInfo* master_info) const;
  void Clear();
  void RequestThetaMatrix(const GetThetaMatrixArgs& get_theta_args,
                          ::artm::ThetaMatrix* theta_matrix) const;
  std::shared_ptr<DataLoaderCacheEntry> FindCacheEntry(const boost::uuids::uuid& batch_uuid) const;
  void UpdateCacheEntry(std::shared_ptr<DataLoaderCacheEntry> cache_entry) const;

 private:
  mutable ThreadSafeCollectionHolder<boost::uuids::uuid, DataLoaderCacheEntry> cache_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_CACHE_MANAGER_H_
