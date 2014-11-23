// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_HELPERS_H_
#define SRC_ARTM_CORE_HELPERS_H_

#include <memory>
#include <string>
#include <vector>

#include "boost/uuid/uuid.hpp"             // uuid class
#include "boost/filesystem.hpp"
#include "boost/thread.hpp"
#include "boost/thread/tss.hpp"

namespace artm {

class Batch;

namespace core {

class Helpers {
 public:
  // Usage: SetThreadName (-1, "MainThread");
  // (thread_id == -1 stands for the current thread)
  static void SetThreadName(int thread_id, const char* thread_name);
};

// This class provides a thread-safe source of float-point random numbers.
// It wraps standard rand_r() function, and calls it with thread-specific seed.
// To store the seed we use boost::thread_specific_ptr.
// For each thread the seed is initialized with an incremental seed (0, 1, 2, etc).
// The initialization is protected with a lock.
// Code example:
//   float random = ThreadSafeRandom::singleton().GenerateFloat();
class ThreadSafeRandom {
 public:
  ThreadSafeRandom() : lock_(), tss_seed_(), seed_(0) {}
  static ThreadSafeRandom& singleton();
  float GenerateFloat();

 private:
  mutable boost::mutex lock_;
  boost::thread_specific_ptr<unsigned int> tss_seed_;
  int seed_;
};

class BatchHelpers {
 public:
  static void CompactBatch(const Batch& batch, Batch* compacted_batch);
  static std::vector<BatchManagerTask> ListAllBatches(const boost::filesystem::path& root);
  static boost::uuids::uuid SaveBatch(const Batch& batch, const std::string& disk_path);

  static void LoadMessage(const std::string& full_filename,
                          ::google::protobuf::Message* message);
  static void LoadMessage(const std::string& filename, const std::string& disk_path,
                          ::google::protobuf::Message* message);
  static void SaveMessage(const std::string& full_filename,
                          const ::google::protobuf::Message& message);
  static void SaveMessage(const std::string& filename, const std::string& disk_path,
                          const ::google::protobuf::Message& message);
  static void PopulateClassId(Batch* batch);
  static bool PopulateThetaMatrixFromCacheEntry(const DataLoaderCacheEntry& cache,
                                                const GetThetaMatrixArgs& get_theta_args,
                                                ::artm::ThetaMatrix* theta_matrix);
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_HELPERS_H_
