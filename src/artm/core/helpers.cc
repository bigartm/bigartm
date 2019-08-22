// Copyright 2017, Additive Regularization of Topic Models.

#include <stdlib.h>

#include <fstream>  // NOLINT
#include <sstream>
#include <thread>  // NOLINT

#include "boost/filesystem.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/random/uniform_real.hpp"
#include "boost/random/variate_generator.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

#include "artm/core/check_messages.h"
#include "artm/core/common.h"
#include "artm/core/helpers.h"
#include "artm/core/exceptions.h"
#include "artm/core/protobuf_helpers.h"
#include "artm/core/token.h"

#if defined(_MSC_VER)
#include <windows.h>  // NOLINT

const DWORD MS_VC_EXCEPTION = 0x406D1388;

#pragma pack(push, 8)
typedef struct tagTHREADNAME_INFO {
  DWORD dwType;      // Must be 0x1000.
  LPCSTR szName;     // Pointer to name (in user addr space).
  DWORD dwThreadID;  // Thread ID (-1=caller thread).
  DWORD dwFlags;     // Reserved for future use, must be zero.
} THREADNAME_INFO;
#pragma pack(pop)

#elif defined(__linux__)

#include <sys/prctl.h>

#endif

namespace artm {
namespace core {

#if defined(_MSC_VER)

// How to: Set a Thread Name in Native Code:
// http://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx
void Helpers::SetThreadName(int thread_id, const char* thread_name) {
  THREADNAME_INFO info;
  info.dwType = 0x1000;
  info.szName = thread_name;
  info.dwThreadID = static_cast<DWORD>(thread_id);
  info.dwFlags = 0;

  /*__try {
    RaiseException( MS_VC_EXCEPTION, 0, sizeof(info)/sizeof(ULONG_PTR), (ULONG_PTR*)&info );  // NOLINT
  }
  __except(EXCEPTION_EXECUTE_HANDLER) {
  }*/
}

#elif defined(__linux__)

// Based on http://stackoverflow.com/questions/778085/how-to-name-a-thread-in-linux
void Helpers::SetThreadName(int thread_id, const char* thread_name) {
  prctl(PR_SET_NAME, thread_name, 0, 0);
}

#else

void Helpers::SetThreadName(int thread_id, const char* thread_name) {
  // Currently not implemented for other systems
}

#endif

std::vector<float> Helpers::GenerateRandomVector(int size, size_t seed) {
  std::vector<float> retval;
  retval.reserve(size);

  boost::mt19937 rng(seed);
  boost::uniform_real<float> u(0.0f, 1.0f);
  boost::variate_generator<boost::mt19937&, boost::uniform_real<float> > gen(rng, u);

  for (int i = 0; i < size; ++i) {
    retval.push_back(gen());
  }

  float sum = 0.0f;
  for (int i = 0; i < size; ++i) {
    sum += retval[i];
  }
  if (sum > 0) {
    for (int i = 0; i < size; ++i) retval[i] /= sum;
  }

  return retval;
}

std::vector<float> Helpers::GenerateRandomVector(int size, const Token& token, int seed) {
  size_t h = 1125899906842597L;  // prime

  if (token.class_id != DefaultClass) {
    for (unsigned i = 0; i < token.class_id.size(); i++) {
      h = 31 * h + token.class_id[i];
    }
  }

  h = 31 * h + 255;  // separate class_id and token

  for (unsigned i = 0; i < token.keyword.size(); i++) {
    h = 31 * h + token.keyword[i];
  }

  if (seed > 0) {
    h = 31 * h + seed;
  }

  return GenerateRandomVector(size, h);
}

// Return the filenames of all files that have the specified extension
// in the specified directory.
std::vector<boost::filesystem::path> Helpers::ListAllBatches(const boost::filesystem::path& root) {
  std::vector<boost::filesystem::path> batches;

  if (boost::filesystem::exists(root) && boost::filesystem::is_directory(root)) {
    boost::filesystem::recursive_directory_iterator it(root);
    boost::filesystem::recursive_directory_iterator endit;
    while (it != endit) {
      if (boost::filesystem::is_regular_file(*it) && it->path().extension() == kBatchExtension) {
        batches.push_back(it->path());
      }
      ++it;
    }
  }
  return batches;
}

boost::uuids::uuid Helpers::SaveBatch(const Batch& batch,
                                      const std::string& disk_path, const std::string& name) {
  if (!batch.has_id()) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Helpers::SaveBatch: batch expecting id"));
  }

  boost::uuids::uuid uuid;
  try {
    uuid = boost::lexical_cast<boost::uuids::uuid>(batch.id());
  } catch (...) {
    BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException("Batch.id", batch.id(), "expecting guid"));
  }

  boost::filesystem::path file(name + kBatchExtension);
  Helpers::SaveMessage(file.string(), disk_path, batch);
  return uuid;
}

void Helpers::LoadMessage(const std::string& filename, const std::string& disk_path,
                          ::google::protobuf::Message* message) {
  boost::filesystem::path full_path =
    boost::filesystem::path(disk_path) / boost::filesystem::path(filename);

  LoadMessage(full_path.string(), message);
}

void Helpers::LoadMessage(const std::string& full_filename,
                          ::google::protobuf::Message* message) {
  std::ifstream fin(full_filename.c_str(), std::ifstream::binary);
  if (!fin.is_open()) {
    BOOST_THROW_EXCEPTION(DiskReadException("Unable to open file " + full_filename));
  }

  message->Clear();
  if (!message->ParseFromIstream(&fin)) {
    BOOST_THROW_EXCEPTION(DiskReadException(
      "Unable to parse protobuf message from " + full_filename));
  }

  fin.close();

  Batch* batch = dynamic_cast<Batch*>(message);
  if ((batch != nullptr) && !batch->has_id()) {
    boost::uuids::uuid uuid;

    try {
      // Attempt to detect UUID based on batche's filename
      std::string filename_only = boost::filesystem::path(full_filename).stem().string();
      uuid = boost::lexical_cast<boost::uuids::uuid>(filename_only);
    } catch (...) { }

    if (uuid.is_nil()) {
      // Otherwise throw the exception
        BOOST_THROW_EXCEPTION(DiskReadException(
          "Unable to detect batch.id or uuid filename in " + full_filename));
    }

    batch->set_id(boost::lexical_cast<std::string>(uuid));
  }

  if (batch != nullptr) {
    FixAndValidateMessage(batch);
  }
}

void Helpers::CreateFolderIfNotExists(const std::string& disk_path) {
  boost::filesystem::path dir(disk_path);
  if (!boost::filesystem::is_directory(dir)) {
    if (!boost::filesystem::create_directory(dir)) {
      BOOST_THROW_EXCEPTION(DiskWriteException("Unable to create folder '" + disk_path + "'"));
    }
  }
}

void Helpers::SaveMessage(const std::string& filename, const std::string& disk_path,
                          const ::google::protobuf::Message& message) {
  CreateFolderIfNotExists(disk_path);
  boost::filesystem::path full_filename = boost::filesystem::path(disk_path) / boost::filesystem::path(filename);
  if (boost::filesystem::exists(full_filename)) {
    LOG(WARNING) << "File already exists: " << full_filename.string();
  }

  SaveMessage(full_filename.string(), message);
}

void Helpers::SaveMessage(const std::string& full_filename,
                          const ::google::protobuf::Message& message) {
  std::ofstream fout(full_filename.c_str(), std::ofstream::binary);
  if (!fout.is_open()) {
    BOOST_THROW_EXCEPTION(DiskReadException("Unable to create file " + full_filename));
  }

  if (!message.SerializeToOstream(&fout)) {
    BOOST_THROW_EXCEPTION(DiskWriteException("Batch has not been serialized to disk."));
  }

  fout.close();
}

bool isZero(float value, float tol) {
  return std::fabs(value) < tol;
}

bool isZero(double value, double tol) {
  return std::fabs(value) < tol;
}

}  // namespace core
}  // namespace artm
