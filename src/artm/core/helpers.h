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

#include "artm/core/common.h"
#include "artm/messages.pb.h"

namespace artm {

class Batch;

namespace core {

class Helpers {
 public:
  // Usage: SetThreadName (-1, "MainThread");
  // (thread_id == -1 stands for the current thread)
  static void SetThreadName(int thread_id, const char* thread_name);
  static std::vector<float> GenerateRandomVector(int size, size_t seed);
  static std::vector<float> GenerateRandomVector(int size, const Token& token, int seed = -1);
};

class BatchHelpers {
 public:
  static void CompactBatch(const Batch& batch, Batch* compacted_batch);
  static std::vector<std::string> ListAllBatches(const boost::filesystem::path& root);
  static boost::uuids::uuid SaveBatch(const Batch& batch, const std::string& disk_path);

  static void LoadMessage(const std::string& full_filename,
                          ::google::protobuf::Message* message);
  static void LoadMessage(const std::string& filename, const std::string& disk_path,
                          ::google::protobuf::Message* message);
  static void SaveMessage(const std::string& full_filename,
                          const ::google::protobuf::Message& message);
  static void SaveMessage(const std::string& filename, const std::string& disk_path,
                          const ::google::protobuf::Message& message);
  static bool PopulateThetaMatrixFromCacheEntry(const DataLoaderCacheEntry& cache,
                                                const GetThetaMatrixArgs& get_theta_args,
                                                ::artm::ThetaMatrix* theta_matrix);
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_HELPERS_H_
