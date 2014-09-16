// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_DATA_LOADER_H_
#define SRC_ARTM_CORE_DATA_LOADER_H_

#include <atomic>
#include <list>
#include <set>
#include <utility>

#include "boost/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"
#include "boost/uuid/uuid.hpp"

#include "artm/messages.pb.h"
#include "artm/core/common.h"
#include "artm/core/template_manager.h"
#include "artm/core/thread_safe_holder.h"

namespace rpcz {
  class application;
}

namespace artm {
namespace core {

class Instance;
class MasterComponentService_Stub;
class Generation;

class DataLoader : boost::noncopyable, public Notifiable {
 public:
  explicit DataLoader(Instance* instance);
  virtual ~DataLoader() {}

  virtual void Callback(std::shared_ptr<const ModelIncrement> model_increment) = 0;
  Instance* instance();

 protected:
  void PopulateDataStreams(const Batch& batch, ProcessorInput* pi);

  Instance* instance_;
};

// DataLoader for local modus operandi
class LocalDataLoader : public DataLoader {
 public:
  explicit LocalDataLoader(Instance* instance);
  virtual ~LocalDataLoader();

  int GetTotalItemsCount() const;
  void AddBatch(const Batch& batch);
  virtual void Callback(std::shared_ptr<const ModelIncrement> model_increment);

  void InvokeIteration(int iterations_count);

  // Returns false if BigARTM is still processing the collection, otherwise true.
  bool WaitIdle(int timeout = -1);
  void DisposeModel(ModelName model_name);
  bool RequestThetaMatrix(ModelName model_name, ::artm::ThetaMatrix* theta_matrix);

 private:
  ThreadSafeHolder<Generation> generation_;

  typedef std::pair<boost::uuids::uuid, ModelName> CacheKey;
  ThreadSafeCollectionHolder<CacheKey, DataLoaderCacheEntry> cache_;

  mutable std::atomic<bool> is_stopping;

  // Keep all threads at the end of class members
  // (because the order of class members defines initialization order;
  // everything else should be initialized before creating threads).
  boost::thread thread_;

  void ThreadFunction();
};

// DataLoader for network modus operandi
class RemoteDataLoader : public DataLoader {
 public:
  explicit RemoteDataLoader(Instance* instance);
  virtual ~RemoteDataLoader();
  virtual void Callback(std::shared_ptr<const ModelIncrement> model_increment);

 private:
  mutable std::atomic<bool> is_stopping;

  // Keep all threads at the end of class members
  // (because the order of class members defines initialization order;
  // everything else should be initialized before creating threads).
  boost::thread thread_;

  void ThreadFunction();
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_DATA_LOADER_H_
