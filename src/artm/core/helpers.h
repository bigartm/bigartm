// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "boost/uuid/uuid.hpp"             // uuid class
#include "boost/filesystem.hpp"
#include "boost/thread.hpp"
#include "boost/thread/tss.hpp"

#include "artm/core/common.h"

namespace artm {
namespace core {

struct Token;

// 'class Helpers' is a utility with several static methods.
class Helpers {
 public:
  // Sets a string for a given thread to simplify debugging.
  // Usage: SetThreadName (-1, "MainThread");
  // (thread_id == -1 stands for the current thread)
  static void SetThreadName(int thread_id, const char* thread_name);

  // Generates random vector using mersenne_twister_engine from boost library.
  // The goal is to ensure that this method is cross-platrofm, e.g. the resulting random vector
  // are the same on Linux, Mac OS and Windows. This is important because
  // the method is used to initialize entries in the phi matrix.
  // For unit-tests it is important that such initialization is deterministic
  // (depends only on the keyword and class_id of the token.
  static std::vector<float> GenerateRandomVector(int size, size_t seed);
  static std::vector<float> GenerateRandomVector(int size, const Token& token, int seed = -1);

  // Lists all batches in a given folder
  static std::vector<boost::filesystem::path> ListAllBatches(const boost::filesystem::path& root);

  // Saves batch to disk
  static boost::uuids::uuid SaveBatch(const Batch& batch,
                                      const std::string& disk_path,
                                      const std::string& name);

  // Loads protobuf message from disk.
  static void LoadMessage(const std::string& full_filename,
                          ::google::protobuf::Message* message);
  static void LoadMessage(const std::string& filename, const std::string& disk_path,
                          ::google::protobuf::Message* message);

  static void CreateFolderIfNotExists(const std::string& disk_path);

  // Saves protobuf message to disk.
  static void SaveMessage(const std::string& full_filename,
                          const ::google::protobuf::Message& message);
  static void SaveMessage(const std::string& filename, const std::string& disk_path,
                          const ::google::protobuf::Message& message);
};

bool isZero(float value, float tol = 1e-16f);
bool isZero(double value, double tol = 1e-16);

}  // namespace core
}  // namespace artm
