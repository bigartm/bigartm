// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/thread/locks.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"

#include "artm/core/common.h"
#include "artm/core/thread_safe_holder.h"

namespace artm {
namespace core {

class Instance;

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
//
// CacheManager can also store the cache as a PhiMatrix.
// Note that this mode might be slower due to lock/guards that prevent several threads
// from calling PhiMatrix::AddToken simultaneously (on the same phi matrix).
// This mode is activated by setting a non-empty MasterModelConfig.ptd_name, indicating a name of ptd matrix.
// To have access to phi matrices CacheManager stores a pointer to the Instance object.
// These are the three "modus operandi" options for CacheManager:
// - disk_path is empty, instance is nullptr --- caching happens in CacheManager::cache_
// - disk_path is not empty, instance is nullptr --- caching happens in CacheManager::cache_,
//   but the actual entries are stored on disk
// - instance is not nullptr and ptd_name is not empty --- chaching happens in PhiMatrix named as ptd_name.
//   (in this case disk_path is ignored).
class CacheManager : boost::noncopyable {
 public:
  explicit CacheManager(const std::string& disk_path, Instance* instance);
  virtual ~CacheManager();

  void RequestMasterComponentInfo(MasterComponentInfo* master_info) const;
  void Clear();
  void RequestThetaMatrix(const GetThetaMatrixArgs& get_theta_args,
                          ::artm::ThetaMatrix* theta_matrix) const;
  std::shared_ptr<ThetaMatrix> FindCacheEntry(const Batch& batch) const;
  void UpdateCacheEntry(const std::string& batch_id, const ThetaMatrix& theta_matrix) const;
  void CopyFrom(const CacheManager& cache_manager);

 private:
  mutable boost::mutex lock_;
  std::string disk_path_;
  Instance* instance_;
  mutable ThreadSafeCollectionHolder<std::string, ThetaCacheEntry> cache_;

  std::shared_ptr<ThetaMatrix> FindCacheEntry(const std::string& batch_id) const;
};

}  // namespace core
}  // namespace artm
