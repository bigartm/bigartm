// Copyright 2014, Additive Regularization of Topic Models.

#include <stdlib.h>

#include <fstream>  // NOLINT
#include <sstream>

#include "boost/filesystem.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

#include "artm/messages.pb.h"
#include "artm/core/common.h"
#include "artm/core/helpers.h"
#include "artm/core/exceptions.h"
#include "artm/core/protobuf_helpers.h"

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

#define ARTM_HELPERS_REPORT_ERROR(error_message)                            \
  if (throw_error) BOOST_THROW_EXCEPTION(InvalidOperation(error_message));  \
  else             LOG(WARNING) << error_message;                           \
  return false;                                                             \

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

void Helpers::Fix(::artm::TopicModel* message) {
  const int token_size = message->token_size();
  if ((message->class_id_size() == 0) && (token_size > 0)) {
    message->mutable_class_id()->Reserve(token_size);
    for (int i = 0; i < token_size; ++i)
      message->add_class_id(::artm::core::DefaultClass);
  }

  if (message->topic_name_size() > 0)
    message->set_topics_count(message->topic_name_size());
}

bool Helpers::Validate(const ::artm::TopicModel& message, bool throw_error) {
  std::stringstream ss;
  const int token_size = message.token_size();
  const bool use_sparse_format = (message.topic_index_size() != 0);
  if ((message.class_id_size() != token_size) ||
      (message.operation_type_size() != token_size) ||
      (message.token_weights_size() != token_size) ||
      (use_sparse_format && (message.topic_index_size() != token_size))) {
    ss << "Inconsistent fields size in TopicModel: "
       << message.token_size() << " vs " << message.class_id_size()
       << " vs " << message.operation_type_size() << " vs " << message.token_weights_size() << ";";
  }

  if (message.topics_count() == 0 || message.topic_name_size() == 0)
    ss << "TopicModel.topic_name_size is empty";
  if (message.topics_count() != message.topic_name_size())
    ss << "Length mismatch in fields TopicModel.topics_count and TopicModel.topic_name";

  for (int i = 0; i < message.token_size(); ++i) {
    bool use_sparse_format_local = use_sparse_format && (message.topic_index(i).value_size() > 0);
    if (use_sparse_format_local) {
      if (message.topic_index(i).value_size() != message.token_weights(i).value_size()) {
        ss << "Length mismatch between TopicModel.topic_index(" << i << ") and TopicModel.token_weights(" << i << ")";
        break;
      }

      bool ok = true;
      for (int topic_index : message.topic_index(i).value()) {
        if (topic_index < 0 || topic_index >= message.topics_count()) {
          ss << "Value " << topic_index << " in message.topic_index(" << i
             << ") is negative or exceeds TopicModel.topics_count";
          ok = false;
          break;
        }
      }

      if (!ok)
        break;
    }

    if (!use_sparse_format) {
      if (message.operation_type(i) == TopicModel_OperationType_Increment ||
          message.operation_type(i) == TopicModel_OperationType_Overwrite) {
        if (message.token_weights(i).value_size() != message.topics_count()) {
          ss << "Length mismatch between TopicModel.topics_count and TopicModel.token_weights(" << i << ")";
          break;
        }
      }
    }
  }

  if (ss.str().empty())
    return true;

  if (throw_error)
    BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
  LOG(WARNING) << ss.str();
  return false;
}

bool Helpers::FixAndValidate(::artm::TopicModel* message, bool throw_error) {
  Fix(message);
  return Validate(*message, throw_error);
}

void Helpers::Fix(::artm::ModelConfig* message) {
  if (message->topic_name_size() == 0) {
    for (int i = 0; i < message->topics_count(); ++i) {
      message->add_topic_name("@topic_" + std::to_string(i));
    }
  } else {
    message->set_topics_count(message->topic_name_size());
  }

  if (message->class_weight_size() == 0) {
    for (int i = 0; i < message->class_id_size(); ++i)
      message->add_class_weight(1.0f);
  }

  if (message->regularizer_tau_size() == 0) {
    for (int i = 0; i < message->regularizer_name_size(); ++i)
      message->add_regularizer_tau(1.0);
  }
}

bool Helpers::Validate(const ::artm::ModelConfig& message, bool throw_error) {
  std::stringstream ss;
  if (message.topics_count() == 0 || message.topic_name_size() == 0)
    ss << "ModelConfig.topic_name() is empty";
  if (message.topics_count() !=  message.topic_name_size())
    ss << "Length mismatch in fields ModelConfig.topics_count and ModelConfig.topic_name";
  if (message.class_weight_size() != message.class_id_size())
    ss << "Length mismatch in fields ModelConfig.class_id and ModelConfig.class_weight";
  if (message.regularizer_name_size() != message.regularizer_tau_size())
    ss << "Length mismatch in fields ModelConfig.regularizer_name_size and ModelConfig.regularizer_tau_size";

  if (ss.str().empty())
    return true;

  if (throw_error)
    BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
  LOG(WARNING) << ss.str();
  return false;
}

bool Helpers::FixAndValidate(::artm::ModelConfig* message, bool throw_error) {
  Fix(message);
  return Validate(*message, throw_error);
}

void Helpers::Fix(::artm::ThetaMatrix* message) {
}

bool Helpers::Validate(const ::artm::ThetaMatrix& message, bool throw_error) {
  std::stringstream ss;
  const int item_size = message.item_id_size();
  const bool has_title = (message.item_title_size() > 0);
  const bool use_sparse_format = (message.topic_index_size() != 0);
  if ((message.item_weights_size() != item_size) ||
      (has_title && (message.item_title_size() != item_size)) ||
      (use_sparse_format && (message.topic_index_size() != item_size))) {
    ss << "Inconsistent fields size in ThetaMatrix: "
       << message.item_id_size() << " vs " << message.item_weights_size()
       << " vs " << message.item_title_size() << " vs " << message.topic_index_size() << ";";
  }

  if (message.topics_count() == 0 || message.topic_name_size() == 0)
    ss << "ThetaMatrix.topic_name_size is empty";
  if (message.topics_count() != message.topic_name_size())
    ss << "Length mismatch in fields ThetaMatrix.topics_count and ThetaMatrix.topic_name";

  for (int i = 0; i < message.item_id_size(); ++i) {
    if (use_sparse_format) {
      if (message.topic_index(i).value_size() != message.item_weights(i).value_size()) {
        ss << "Length mismatch between ThetaMatrix.topic_index(" << i << ") and ThetaMatrix.item_weights(" << i << ")";
        break;
      }

      bool ok = true;
      for (int topic_index : message.topic_index(i).value()) {
        if (topic_index < 0 || topic_index >= message.topics_count()) {
          ss << "Value " << topic_index << " in message.topic_index(" << i
             << ") is negative or exceeds ThetaMatrix.topics_count";
          ok = false;
          break;
        }
      }

      if (!ok)
        break;
    }
  }

  if (ss.str().empty())
    return true;

  if (throw_error)
    BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
  LOG(WARNING) << ss.str();
  return false;
}

bool Helpers::FixAndValidate(::artm::ThetaMatrix* message, bool throw_error) {
  Fix(message);
  return Validate(*message, throw_error);
}

void Helpers::Fix(::artm::Batch* message) {
  if (message->class_id_size() == 0) {
    for (int i = 0; i < message->token_size(); ++i) {
      message->add_class_id(DefaultClass);
    }
  }
}

bool Helpers::Validate(const ::artm::Batch& message, bool throw_error) {
  std::stringstream ss;
  if (message.has_id()) {
    try {
      boost::lexical_cast<boost::uuids::uuid>(message.id());
    }
    catch (...) {
      ss << "Batch.id must be GUID, got: " << message.id();
      ARTM_HELPERS_REPORT_ERROR(ss.str())
    }
  } else {
    ARTM_HELPERS_REPORT_ERROR("Batch.id is not specified");
  }

  if (message.class_id_size() != message.token_size()) {
    ss << "Length mismatch in fields Batch.class_id and Batch.token, batch.id = " << message.id();
    ARTM_HELPERS_REPORT_ERROR(ss.str());
  }

  for (int item_id = 0; item_id < message.item_size(); ++item_id) {
    for (const Field& field : message.item(item_id).field()) {
      if (field.token_count_size() != field.token_id_size()) {
        ss << "Length mismatch in field Batch.item(" << item_id << ").token_count and token_id; ";
        break;
      }

      for (int token_index = 0; token_index < field.token_count_size(); token_index++) {
        int token_id = field.token_id(token_index);
        int token_count = field.token_count(token_index);
        if (token_id < 0 || token_id >= message.token_size()) {
          ss << "Value " << token_id << " in Batch.Item(" << item_id
             << ").token_id is negative or exceeds Batch.token_size";
          ARTM_HELPERS_REPORT_ERROR(ss.str());
        }
      }
    }
  }

  if (ss.str().empty())
    return true;

  if (throw_error)
    BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
  LOG(WARNING) << ss.str();
  return false;
}

bool Helpers::FixAndValidate(::artm::Batch* message, bool throw_error) {
  Fix(message);
  return Validate(*message, throw_error);
}

std::string Helpers::Describe(const ::artm::ModelConfig& message) {
  std::stringstream ss;
  ss << "ModelConfig";
  ss << ": name=" << message.name();
  ss << ", topics_count=" << message.topics_count();
  ss << ", topic_name_size=" << message.topic_name_size();
  ss << ", enabled=" << (message.enabled() ? "yes" : "no");
  ss << ", inner_iterations_count=" << message.inner_iterations_count();
  ss << ", field_name=" << message.field_name();
  ss << ", stream_name=" << message.stream_name();
  ss << ", reuse_theta=" << (message.reuse_theta() ? "yes" : "no");
  for (int i = 0; i < message.regularizer_name_size(); ++i)
    ss << ", regularizer=(" << message.regularizer_name(i) << ":" << message.regularizer_tau(i) << ")";
  for (int i = 0; i < message.class_id_size(); ++i)
    ss << ", class=(" << message.class_id(i) << ":" << message.class_weight(i) << ")";
  ss << ", use_sparse_bow=" << (message.use_sparse_bow() ? "yes" : "no");
  ss << ", use_random_theta=" << (message.use_random_theta() ? "yes" : "no");
  ss << ", use_new_tokens=" << (message.use_new_tokens() ? "yes" : "no");
  return ss.str();
}

std::string Helpers::Describe(const ::artm::MasterComponentConfig& message) {
  std::stringstream ss;
  ss << "MasterComponentConfig";
  ss << ": modus_operandi=" << message.modus_operandi();
  ss << ", disk_path=" << message.disk_path();
  ss << ", stream_size=" << message.stream_size();
  ss << ", compact_batches=" << (message.compact_batches() ? "yes" : "no");
  ss << ", cache_theta=" << (message.cache_theta() ? "yes" : "no");
  ss << ", processors_count=" << message.processors_count();
  ss << ", processor_queue_max_size=" << message.processor_queue_max_size();
  ss << ", merger_queue_max_size=" << message.merger_queue_max_size();
  ss << ", score_config_size=" << message.score_config_size();
  if (message.modus_operandi() == MasterComponentConfig_ModusOperandi_Network) {
    ss << ", create_endpoint=" << message.create_endpoint();
    ss << ", connect_endpoint=" << message.connect_endpoint();
    ss << ", node_connect_endpoint_size=" << message.node_connect_endpoint_size();
    ss << ", communication_timeout=" << message.communication_timeout();
  }

  ss << ", disk_cache_path" << message.disk_cache_path();
  return ss.str();
}

std::vector<float> Helpers::GenerateRandomVector(int size, size_t seed) {
  std::vector<float> retval;
  retval.reserve(size);

#if defined(_WIN32) || defined(_WIN64)
  // http://msdn.microsoft.com/en-us/library/aa272875(v=vs.60).aspx
  // rand() is thread-safe on Windows when linked with LIBCMT.LIB

  srand(seed);
  for (int i = 0; i < size; ++i) {
    retval.push_back(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));  // NOLINT
  }
#else
  unsigned int int_seed = static_cast<unsigned int>(seed);
  for (int i = 0; i < size; ++i) {
    retval.push_back(static_cast<float>(rand_r(&int_seed)) / static_cast<float>(RAND_MAX));
  }
#endif

  float sum = 0.0f;
  for (int i = 0; i < size; ++i) sum += retval[i];
  if (sum > 0) {
    for (int i = 0; i < size; ++i) retval[i] /= sum;
  }

  return retval;
}

// Return the filenames of all files that have the specified extension
// in the specified directory.
std::vector<BatchManagerTask> BatchHelpers::ListAllBatches(const boost::filesystem::path& root) {
  std::vector<BatchManagerTask> uuids;

  if (boost::filesystem::exists(root) && boost::filesystem::is_directory(root)) {
    boost::filesystem::recursive_directory_iterator it(root);
    boost::filesystem::recursive_directory_iterator endit;
    while (it != endit) {
      if (boost::filesystem::is_regular_file(*it) && it->path().extension() == kBatchExtension) {
        std::string filename = it->path().filename().stem().string();
        boost::uuids::uuid uuid = boost::uuids::nil_uuid();
        try { uuid = boost::lexical_cast<boost::uuids::uuid>(filename); } catch (...) {}
        if (uuid.is_nil()) {
          uuid = boost::uuids::random_generator()();
          LOG(INFO) << "Use " << uuid << " as uuid for batch " << it->path().string();
        }

        uuids.push_back(BatchManagerTask(uuid, it->path().string()));
      }
      ++it;
    }
  }
  return uuids;
}

boost::uuids::uuid BatchHelpers::SaveBatch(const Batch& batch,
                                           const std::string& disk_path) {
  boost::uuids::uuid uuid;
  if (batch.has_id()) {
    try {
      uuid = boost::lexical_cast<boost::uuids::uuid>(batch.id());
    } catch (...) {
      BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException("Batch.id", batch.id(), "expecting guid"));
    }
  } else {
    uuid = boost::uuids::random_generator()();
  }

  boost::filesystem::path file(boost::lexical_cast<std::string>(uuid) + kBatchExtension);
  SaveMessage(file.string(), disk_path, batch);
  return uuid;
}

void BatchHelpers::CompactBatch(const Batch& batch, Batch* compacted_batch) {
  if (batch.has_description()) compacted_batch->set_description(batch.description());
  if (batch.has_id()) compacted_batch->set_id(batch.id());

  std::vector<int> orig_to_compacted_id_map(batch.token_size(), -1);
  int compacted_dictionary_size = 0;

  bool has_class_id = (batch.class_id_size() > 0);
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
        if (has_class_id && (token_id >= batch.class_id_size()))
          BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
            "field.token_id", token_id, "Too few entries in batch.class_id field"));

        if (orig_to_compacted_id_map[token_id] == -1) {
          orig_to_compacted_id_map[token_id] = compacted_dictionary_size++;
          compacted_batch->add_token(batch.token(token_id));
          if (has_class_id)
            compacted_batch->add_class_id(batch.class_id(token_id));
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
    } catch (...) {}

    if (uuid.is_nil()) {
      // Otherwise generate new random UUID
      uuid = boost::uuids::random_generator()();
    }

    batch->set_id(boost::lexical_cast<std::string>(uuid));
  }

  if (batch != nullptr)
    Helpers::FixAndValidate(batch);
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

bool BatchHelpers::PopulateThetaMatrixFromCacheEntry(
    const DataLoaderCacheEntry& cache,
    const GetThetaMatrixArgs& get_theta_args,
    ::artm::ThetaMatrix* theta_matrix) {
  if (get_theta_args.topic_index_size() != 0 && get_theta_args.topic_name_size() != 0)
    BOOST_THROW_EXCEPTION(InvalidOperation(
    "GetThetaMatrixArgs.topic_name and GetThetaMatrixArgs.topic_index must not be used together"));

  auto& args_model_name = get_theta_args.model_name();
  auto& args_topic_name = get_theta_args.topic_name();
  auto& args_topic_index = get_theta_args.topic_index();
  const bool use_sparse_format = get_theta_args.use_sparse_format();

  std::vector<int> topics_to_use;
  if (args_topic_index.size() > 0) {
    for (int i = 0; i < args_topic_index.size(); ++i) {
      int topic_index = args_topic_index.Get(i);
      if (topic_index < 0 || topic_index >= cache.topic_name_size()) {
        std::stringstream ss;
        ss << "GetThetaMatrixArgs.topic_index[" << i << "] == " << topic_index << " is out of range.";
        BOOST_THROW_EXCEPTION(artm::core::InvalidOperation(ss.str()));
      }
      topics_to_use.push_back(topic_index);
    }
  } else if (args_topic_name.size() != 0) {
    for (int i = 0; i < args_topic_name.size(); ++i) {
      int topic_index = repeated_field_index_of(cache.topic_name(), args_topic_name.Get(i));
      if (topic_index == -1) {
        std::stringstream ss;
        ss << "GetThetaMatrixArgs.topic_name[" << i << "] == " << args_topic_name.Get(i)
           << " does not exist in ModelConfig.topic_name";
        BOOST_THROW_EXCEPTION(artm::core::InvalidOperation(ss.str()));
      }

      assert(topic_index >= 0 && topic_index < cache.topic_name_size());
      topics_to_use.push_back(topic_index);
    }
  } else {  // use all topics
    assert(cache.topic_name_size() > 0);
    for (int i = 0; i < cache.topic_name_size(); ++i)
      topics_to_use.push_back(i);
  }

  // Populate topics_count and topic_name fields in the resulting message
  ::google::protobuf::RepeatedPtrField< ::std::string> result_topic_name;
  for (int topic_index : topics_to_use)
    result_topic_name.Add()->assign(cache.topic_name(topic_index));

  if (!theta_matrix->has_model_name()) {
    // Assign
    theta_matrix->set_model_name(args_model_name);
    theta_matrix->set_topics_count(result_topic_name.size());
    assert(theta_matrix->topic_name_size() == 0);
    for (TopicName topic_name : result_topic_name)
      theta_matrix->add_topic_name(topic_name);
  } else {
    // Verify
    if (theta_matrix->model_name() != args_model_name)
      BOOST_THROW_EXCEPTION(artm::core::InternalError("theta_matrix->model_name() != args_model_name"));
    if (theta_matrix->topics_count() != result_topic_name.size())
      BOOST_THROW_EXCEPTION(artm::core::InternalError("theta_matrix->topics_count() != result_topic_name.size()"));
    for (int i = 0; i < theta_matrix->topic_name_size(); ++i) {
      if (theta_matrix->topic_name(i) != result_topic_name.Get(i))
        BOOST_THROW_EXCEPTION(artm::core::InternalError("theta_matrix->topic_name(i) != result_topic_name.Get(i)"));
    }
  }

  bool has_title = (cache.item_title_size() == cache.item_id_size());
  for (int item_index = 0; item_index < cache.item_id_size(); ++item_index) {
    theta_matrix->add_item_id(cache.item_id(item_index));
    if (has_title) theta_matrix->add_item_title(cache.item_title(item_index));
    ::artm::FloatArray* theta_vec = theta_matrix->add_item_weights();

    const artm::FloatArray& item_theta = cache.theta(item_index);
    if (!use_sparse_format) {
      for (int topic_index : topics_to_use)
        theta_vec->add_value(item_theta.value(topic_index));
    } else {
      ::artm::IntArray* sparse_topic_index = theta_matrix->add_topic_index();
      for (int topics_to_use_index = 0; topics_to_use_index < topics_to_use.size(); topics_to_use_index++) {
        int topic_index = topics_to_use[topics_to_use_index];
        float value = item_theta.value(topic_index);
        if (value >= get_theta_args.eps()) {
          theta_vec->add_value(item_theta.value(topic_index));
          sparse_topic_index->add_value(topics_to_use_index);
        }
      }
    }
  }

  return true;
}


}  // namespace core
}  // namespace artm
