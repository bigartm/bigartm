// Copyright 2015, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_CACHE_MANAGER_H_
#define SRC_ARTM_CORE_CACHE_MANAGER_H_

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/utility.hpp"
#include "boost/uuid/uuid.hpp"

#include "artm/messages.pb.h"
#include "artm/core/common.h"
#include "artm/core/internals.pb.h"
#include "artm/core/thread_safe_holder.h"

namespace artm {
namespace core {

class CacheManager : boost::noncopyable {
 public:
  explicit CacheManager();
  virtual ~CacheManager();

  void DisposeModel(ModelName model_name);
  bool RequestThetaMatrix(const GetThetaMatrixArgs& get_theta_args,
                          ::artm::ThetaMatrix* theta_matrix) const;
  std::shared_ptr<DataLoaderCacheEntry> FindCacheEntry(const boost::uuids::uuid& batch_uuid,
                                                       const ModelName& model_name) const;
  void UpdateCacheEntry(std::shared_ptr<DataLoaderCacheEntry> cache_entry) const;

 private:
  typedef std::pair<boost::uuids::uuid, ModelName> CacheKey;
  mutable ThreadSafeCollectionHolder<CacheKey, DataLoaderCacheEntry> cache_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_CACHE_MANAGER_H_
