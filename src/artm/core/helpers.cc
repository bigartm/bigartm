// Copyright 2014, Additive Regularization of Topic Models.

#include <stdlib.h>

#include <fstream>  // NOLINT

#include "boost/filesystem.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

#include "artm/messages.pb.h"
#include "artm/core/common.h"
#include "artm/core/helpers.h"
#include "artm/core/exceptions.h"

#if defined(_WIN32) || defined(_WIN64)
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

#if defined(_WIN32) || defined(_WIN64)

// How to: Set a Thread Name in Native Code:
// http://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx
void Helpers::SetThreadName(int thread_id, const char* thread_name) {
  THREADNAME_INFO info;
  info.dwType = 0x1000;
  info.szName = thread_name;
  info.dwThreadID = static_cast<DWORD>(thread_id);
  info.dwFlags = 0;

  __try {
    RaiseException( MS_VC_EXCEPTION, 0, sizeof(info)/sizeof(ULONG_PTR), (ULONG_PTR*)&info );  // NOLINT
  }
  __except(EXCEPTION_EXECUTE_HANDLER) {
  }
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


ThreadSafeRandom& ThreadSafeRandom::singleton() {
  static ThreadSafeRandom instance;
  return instance;
}

float ThreadSafeRandom::GenerateFloat() {
  if (!tss_seed_.get()) {
    boost::lock_guard<boost::mutex> guard(lock_);
    srand(seed_);

    tss_seed_.reset(new unsigned int);
    *tss_seed_ = seed_;
    seed_++;
  }

#if defined(_WIN32) || defined(_WIN64)
  // http://msdn.microsoft.com/en-us/library/aa272875(v=vs.60).aspx
  // rand() is thread-safe on Windows when linked with LIBCMT.LIB
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);  // NOLINT
#else
  return static_cast<float>(rand_r(tss_seed_.get())) / static_cast<float>(RAND_MAX);
#endif
}

// Return the filenames of all files that have the specified extension
// in the specified directory.
std::vector<boost::uuids::uuid> BatchHelpers::ListAllBatches(const boost::filesystem::path& root) {
  std::vector<boost::uuids::uuid> uuids;

  if (boost::filesystem::exists(root) && boost::filesystem::is_directory(root)) {
    boost::filesystem::recursive_directory_iterator it(root);
    boost::filesystem::recursive_directory_iterator endit;
    while (it != endit) {
      if (boost::filesystem::is_regular_file(*it) && it->path().extension() == kBatchExtension) {
        std::string filename = it->path().filename().stem().string();
        boost::uuids::uuid uuid = boost::uuids::string_generator()(filename);
        if (uuid.is_nil()) {
          LOG(WARNING) << "Unable to convert filename " << filename << " to uuid.";
          continue;
        }

        uuids.push_back(uuid);
      }
      ++it;
    }
  }
  return uuids;
}

std::shared_ptr<Batch> BatchHelpers::LoadBatch(const boost::uuids::uuid& uuid,
                                                     const std::string& disk_path) {
  Batch* batch = new Batch();
  std::shared_ptr<Batch> batch_ptr(batch);
  boost::filesystem::path file(boost::lexical_cast<std::string>(uuid) + kBatchExtension);
  LoadMessage(file.string(), disk_path, batch);
  PopulateClassId(batch_ptr.get());
  return batch_ptr;
}

boost::uuids::uuid BatchHelpers::SaveBatch(const Batch& batch,
                                           const std::string& disk_path) {
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  boost::filesystem::path file(boost::lexical_cast<std::string>(uuid) + kBatchExtension);
  SaveMessage(file.string(), disk_path, batch);
  return uuid;
}

void BatchHelpers::CompactBatch(const Batch& batch, Batch* compacted_batch) {
  std::vector<int> orig_to_compacted_id_map(batch.token_size(), -1);
  int compacted_dictionary_size = 0;

  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    auto item = batch.item(item_index);
    auto compacted_item = compacted_batch->add_item();
    compacted_item->CopyFrom(item);

    for (int field_index = 0; field_index < item.field_size(); ++field_index) {
      auto field = item.field(field_index);
      auto compacted_field = compacted_item->mutable_field(field_index);

      for (int token_index = 0; token_index < field.token_id_size(); ++token_index) {
        int token_id = field.token_id(token_index);
        if (token_id < 0 || token_id >= batch.token_size())
          BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException("field.token_id", token_id));

        if (orig_to_compacted_id_map[token_id] == -1) {
          orig_to_compacted_id_map[token_id] = compacted_dictionary_size++;
          compacted_batch->add_token(batch.token(token_id));
        }

        compacted_field->set_token_id(token_index, orig_to_compacted_id_map[token_id]);
      }
    }
  }
}

void BatchHelpers::LoadMessage(const std::string& filename, const std::string& disk_path,
                               ::google::protobuf::Message* message) {
  boost::filesystem::path full_path =
    boost::filesystem::path(disk_path) / boost::filesystem::path(filename);

  LoadMessage(full_path.string(), message);
}

void BatchHelpers::LoadMessage(const std::string& full_filename,
                               ::google::protobuf::Message* message) {
  std::ifstream fin(full_filename.c_str(), std::ifstream::binary);
  if (!fin.is_open())
    BOOST_THROW_EXCEPTION(DiskReadException("Unable to open file " + full_filename));

  if (!message->ParseFromIstream(&fin)) {
    BOOST_THROW_EXCEPTION(DiskReadException(
      "Unable to parse protobuf message from " + full_filename));
  }

  fin.close();
}

void BatchHelpers::SaveMessage(const std::string& filename, const std::string& disk_path,
                               const ::google::protobuf::Message& message) {
  boost::filesystem::path dir(disk_path);
  if (!boost::filesystem::is_directory(dir)) {
    if (!boost::filesystem::create_directory(dir))
      BOOST_THROW_EXCEPTION(DiskWriteException("Unable to create folder '" + disk_path + "'"));
  }

  boost::filesystem::path full_filename = dir / boost::filesystem::path(filename);
  if (boost::filesystem::exists(full_filename))
    BOOST_THROW_EXCEPTION(DiskWriteException("File already exists: " + full_filename.string()));

  SaveMessage(full_filename.string(), message);
}

void BatchHelpers::SaveMessage(const std::string& full_filename,
                               const ::google::protobuf::Message& message) {
  std::ofstream fout(full_filename.c_str(), std::ofstream::binary);
  if (!fout.is_open())
    BOOST_THROW_EXCEPTION(DiskReadException("Unable to create file " + full_filename));

  if (!message.SerializeToOstream(&fout)) {
    BOOST_THROW_EXCEPTION(DiskWriteException("Batch has not been serialized to disk."));
  }

  fout.close();
}

void BatchHelpers::PopulateClassId(Batch* batch) {
  if (batch->class_id_size() != batch->token_size()) {
    if (batch->class_id_size() != 0) {
      // ToDo(alfrey): log the ID of the batch
      LOG(ERROR) << "Field batch.class_id must have the same length as field batch.token. "
                 << "Setting '@DefaultClass' label for all tokens.";
    }

    batch->clear_class_id();
    for (int i = 0; i < batch->token_size(); ++i) {
      batch->add_class_id(DefaultClass);
    }
  }
}

}  // namespace core
}  // namespace artm
