// Copyright 2015, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_CACHE_MANAGER_H_
#define SRC_ARTM_CORE_CACHE_MANAGER_H_

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/utility.hpp"

#include "artm/core/common.h"
#include "artm/core/thread_safe_holder.h"

namespace artm {
namespace core {

class ThetaCacheEntry : boost::noncopyable {
 public:
  ThetaCacheEntry();
  ~ThetaCacheEntry();

  std::shared_ptr<ThetaMatrix> theta_matrix() { return theta_matrix_; }
  const std::string& filename() const { return filename_; }
  std::string* mutable_filename() { return &filename_; }

 private:
  std::shared_ptr<ThetaMatrix> theta_matrix_;
  std::string filename_;
};

// CacheManager class is responsible for caching ThetaMatrix in between calls to different APIs.
// This class is used when the user calls FitOffline / FitOnline / Transfor to store the resulting theta matrix.
// (at least when theta_matrix_type is set to ThetaMatrixType_Cache).
// Later user may retrieve the data from CacheManager via calls to ArtmRequestThetaMatrix.
// The cache is organized as a set of entries, each entry associated with a single batch.
// The key in the cache corresponds to 'batch.id' field.
class CacheManager : boost::noncopyable {
 public:
  explicit CacheManager(const std::string& disk_path);
  virtual ~CacheManager();

  void RequestMasterComponentInfo(MasterComponentInfo* master_info) const;
  void Clear();
  void RequestThetaMatrix(const GetThetaMatrixArgs& get_theta_args,
                          ::artm::ThetaMatrix* theta_matrix) const;
  std::shared_ptr<ThetaMatrix> FindCacheEntry(const std::string& batch_id) const;
  void UpdateCacheEntry(const std::string& batch_id, const ThetaMatrix& theta_matrix) const;

 private:
  std::string disk_path_;
  mutable ThreadSafeCollectionHolder<std::string, ThetaCacheEntry> cache_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_CACHE_MANAGER_H_
