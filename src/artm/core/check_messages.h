// Copyright 2015, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_CHECK_MESSAGES_H_
#define SRC_ARTM_CORE_CHECK_MESSAGES_H_

#include <string>

#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"

#include "glog/logging.h"

#include "artm/messages.pb.h"
#include "artm/core/common.h"
#include "artm/core/exceptions.h"

namespace artm {
namespace core {

template <typename T>
inline bool FixAndValidateMessage(T* message, bool throw_error = true);

template <typename T>
inline bool ValidateMessage(const T& message, bool throw_error = true);

///////////////////////////////////////////////////////////////////////////////////////////////////
// DescribeErrors routines
// This method is required for all messages that go through c_interface.
///////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string DescribeErrors(const ::artm::ModelConfig& message) {
  std::stringstream ss;
  if (message.topics_count() == 0 || message.topic_name_size() == 0)
    ss << "ModelConfig.topic_name() is empty";
  if (message.topics_count() != message.topic_name_size())
    ss << "Length mismatch in fields ModelConfig.topics_count and ModelConfig.topic_name";
  if (message.class_weight_size() != message.class_id_size())
    ss << "Length mismatch in fields ModelConfig.class_id and ModelConfig.class_weight";
  return ss.str();
}

inline std::string DescribeErrors(const ::artm::TopicModel& message) {
  std::stringstream ss;

  const bool has_topic_data = (message.topics_count() != 0 || message.topic_name_size() != 0);
  const bool has_token_data = (message.class_id_size() != 0 || message.token_size() != 0);
  const bool has_bulk_data = (message.token_weights_size() != 0 || message.operation_type_size() != 0);
  const bool has_sparse_format = has_bulk_data && (message.topic_index_size() != 0);

  if (has_topic_data) {
    if (message.topics_count() != message.topic_name_size())
      ss << "Length mismatch in fields TopicModel.topics_count and TopicModel.topic_name";
  }

  if (has_token_data) {
    if (message.class_id_size() != message.token_size())
      ss << "Inconsistent fields size in TopicModel.token and TopicModel.class_id: "
         << message.token_size() << " vs " << message.class_id_size();
  }

  if (has_bulk_data && !has_topic_data)
    ss << "TopicModel.topic_name_size is empty";
  if (has_bulk_data && !has_token_data)
    ss << "TopicModel.token_size is empty";

  if (has_bulk_data) {
    if ((message.operation_type_size() != message.token_size()) ||
      (message.token_weights_size() != message.token_size()) ||
      (has_sparse_format && (message.topic_index_size() != message.token_size()))) {
      ss << "Inconsistent fields size in TopicModel: "
        << message.token_size() << " vs " << message.class_id_size()
        << " vs " << message.operation_type_size() << " vs " << message.token_weights_size() << ";";
    }

    for (int i = 0; i < message.token_size(); ++i) {
      bool has_sparse_format_local = has_sparse_format && (message.topic_index(i).value_size() > 0);
      if (has_sparse_format_local) {
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

      if (!has_sparse_format) {
        if (message.operation_type(i) == TopicModel_OperationType_Increment ||
            message.operation_type(i) == TopicModel_OperationType_Overwrite) {
          if (message.token_weights(i).value_size() != message.topics_count()) {
            ss << "Length mismatch between TopicModel.topics_count and TopicModel.token_weights(" << i << ")";
            break;
          }
        }
      }
    }
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::ThetaMatrix& message) {
  std::stringstream ss;
  const int item_size = message.item_id_size();
  const bool has_title = (message.item_title_size() > 0);
  const bool has_sparse_format = (message.topic_index_size() != 0);
  if ((message.item_weights_size() != item_size) ||
      (has_title && (message.item_title_size() != item_size)) ||
      (has_sparse_format && (message.topic_index_size() != item_size))) {
    ss << "Inconsistent fields size in ThetaMatrix: "
       << message.item_id_size() << " vs " << message.item_weights_size()
       << " vs " << message.item_title_size() << " vs " << message.topic_index_size() << ";";
  }

  if (message.topics_count() == 0 || message.topic_name_size() == 0)
    ss << "ThetaMatrix.topic_name_size is empty";
  if (message.topics_count() != message.topic_name_size())
    ss << "Length mismatch in fields ThetaMatrix.topics_count and ThetaMatrix.topic_name";

  for (int i = 0; i < message.item_id_size(); ++i) {
    if (has_sparse_format) {
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

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::Batch& message) {
  std::stringstream ss;
  if (message.has_id()) {
    try {
      boost::lexical_cast<boost::uuids::uuid>(message.id());
    }
    catch (...) {
      ss << "Batch.id must be GUID, got: " << message.id();
      return ss.str();
    }
  } else {
    ss << "Batch.id is not specified";
    return ss.str();
  }

  const bool has_tokens = (message.token_size() > 0);
  if (!has_tokens && (message.class_id_size() > 0)) {
    ss << "Empty Batch.token require that Batch.class_id must also be empty, batch.id = " << message.id();
    return ss.str();
  }

  if (has_tokens && (message.class_id_size() != message.token_size())) {
    ss << "Length mismatch in fields Batch.class_id and Batch.token, batch.id = " << message.id();
    return ss.str();
  }

  for (int item_id = 0; item_id < message.item_size(); ++item_id) {
    for (const Field& field : message.item(item_id).field()) {
      if (field.token_count_size() != 0) {
        ss << "Field.token_count field is deprecated. Use Field.token_weight instead; ";
        break;
      }

      if (field.token_weight_size() != field.token_id_size()) {
        ss << "Length mismatch in field Batch.item(" << item_id << ").token_weight and token_id; ";
        break;
      }

      for (int token_index = 0; token_index < field.token_id_size(); token_index++) {
        int token_id = field.token_id(token_index);
        if ((token_id < 0) || (has_tokens && (token_id >= message.token_size()))) {
          ss << "Value " << token_id << " in Batch.Item(" << item_id
             << ").token_id is negative or exceeds Batch.token_size";
          return ss.str();
        }
      }
    }
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::GetThetaMatrixArgs& message) {
  std::stringstream ss;
  if (message.has_batch())
    ss << DescribeErrors(message.batch());

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::GetScoreValueArgs& message) {
  std::stringstream ss;

  if (message.has_batch())
    ss << DescribeErrors(message.batch());

  if (!message.has_model_name() || message.model_name().empty())
    ss << "GetScoreValueArgs.model_name is missing; ";
  if (!message.has_score_name() || message.score_name().empty())
    ss << "GetScoreValueArgs.score_name is missing; ";

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::MasterComponentConfig& message) {
  std::stringstream ss;

  if (message.processor_queue_max_size() <= 0)
    ss << "MasterComponentConfig.processor_queue_max_size == "
       << message.processor_queue_max_size() << " is invalid; ";

  if (message.merger_queue_max_size() <= 0)
    ss << "MasterComponentConfig.merger_queue_max_size == "
       << message.merger_queue_max_size() << " is invalid; ";

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::MasterModelConfig& message) {
  std::stringstream ss;

  if (message.class_weight_size() != message.class_id_size())
    ss << "Length mismatch in fields MasterModelConfig.class_id and MasterModelConfig.class_weight; ";

  if (message.inner_iterations_count() < 0)
    ss << "Field MasterModelConfig.inner_iterations_count must be non-negative; ";

  for (int i = 0; i < message.regularizer_config_size(); ++i) {
    const RegularizerConfig& config = message.regularizer_config(i);
    if (!config.has_tau())
      ss << "Field MasterModelConfig.RegularizerConfig.tau must not be empty "
         << "(regularizer name: " << config.name() << "); ";
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::FitOfflineMasterModelArgs& message) {
  std::stringstream ss;

  if (message.batch_filename_size() != message.batch_weight_size())
    ss << "Length mismatch in fields FitOfflineMasterModelArgs.batch_filename "
       << "and FitOfflineMasterModelArgs.batch_weight; ";

  if (message.passes() <= 0)
    ss << "FitOfflineMasterModelArgs.passes() must be a positive number";

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::FitOnlineMasterModelArgs& message) {
  std::stringstream ss;

  if (message.batch_filename_size() == 0)
    ss << "Fields FitOnlineMasterModelArgs.batch_filename must not be empty; ";

  if (message.batch_filename_size() != message.batch_weight_size())
    ss << "Length mismatch in fields FitOnlineMasterModelArgs.batch_filename "
    << "and FitOnlineMasterModelArgs.batch_weight; ";

  if (message.update_after_size() == 0)
    ss << "Field FitOnlineMasterModelArgs.update_after must not be empty; ";

  if (message.update_after_size() != message.apply_weight_size() ||
      message.update_after_size() != message.decay_weight_size()) {
    ss << "Length mismatch in fields FitOnlineMasterModelArgs.update_after, "
       << "FitOnlineMasterModelArgs.apply_weight and FitOnlineMasterModelArgs.decay_weight; ";
  }

  for (int i = 0; i < message.update_after_size(); i++) {
    int value = message.update_after(i);
    if (value <= 0) {
      ss << "FitOnlineMasterModelArgs.update_after[" << i << "] == " << value
         << ", expected value must be greater than zero; ";
      break;
    }
    if (value > message.batch_filename_size()) {
      ss << "FitOnlineMasterModelArgs.update_after[" << i << "] == " << value
         << ", expected value must not exceed FitOnlineMasterModelArgs.batch_filename_size(); ";
      break;
    }
    if ((i > 0) && (message.update_after(i) <= message.update_after(i - 1))) {
      ss << "FitOnlineMasterModelArgs.update_after[" << i << "] "
         << "is less than previous value; expect strictly increasing sequence; ";
      break;
    }
    if ((i + 1) == message.update_after_size()) {
      if (message.update_after(i) != message.batch_filename_size()) {
        ss << "Last element in FitOnlineMasterModelArgs.update_after is " << message.update_after(i) << ", "
           << "expected value is FitOnlineMasterModelArgs.batch_filename_size(), which was "
           << message.batch_filename_size() << "; ";
        break;
      }
    }
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::TransformMasterModelArgs& message) {
  std::stringstream ss;

  if (message.batch_filename_size() == 0 && message.batch_size() == 0)
    ss << "Either TransformMasterModelArgs.batch_filename or TransformMasterModelArgs.batch must be specified; ";
  if (message.batch_filename_size() != 0 && message.batch_size() != 0)
    ss << "Only one of TransformMasterModelArgs.batch_filename, "
       << "TransformMasterModelArgs.batch must be specified; ";

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::InitializeModelArgs& message) {
  std::stringstream ss;

  if (message.topics_count() != 0 || message.topic_name_size() != 0) {
    if (message.topics_count() != message.topic_name_size())
      ss << "Length mismatch in fields InitializeModelArgs.topics_count and InitializeModelArgs.topic_name";
  }

  if (!message.has_model_name()) {
    ss << "InitializeModelArgs.model_name is not defined; ";
  }

  if (!message.has_dictionary_name()) {
    ss << "InitializeModelArgs.dictionary_name is not defined; ";
  }

  if (message.has_source_type() || message.has_disk_path() ||
      message.filter_size() || message.batch_filename_size()) {
    ss << "InitializeModelArgs has no longer support source types (using only dictionary). ";
    ss << "Fields 'disk_path' and 'batch_filename' are deprecated. ";
    ss << "Also it doesn't proceed filtering (use ArtmFilterDictionary())";
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::FilterDictionaryArgs& message) {
  std::stringstream ss;

  if (!message.has_dictionary_name())
    ss << "FilterDictionaryArgs has no dictionary name; ";

  if (!message.has_dictionary_target_name())
     ss << "FilterDictionaryArgs has no target dictionary name; ";

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::CollectionParserConfig& message) {
  std::stringstream ss;

  if (message.cooccurrence_token_size() || message.has_gather_cooc() ||
      message.cooccurrence_class_id_size() || message.has_use_symmetric_cooc_values()) {
    ss << "Collection parser no longer support gathering dictionary and cooc data. ";
    ss << "Use ArtmParseCollection() and then ArtmGatherDictionary() functions";
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::GatherDictionaryArgs& message) {
  std::stringstream ss;

  if (!message.has_dictionary_target_name())
    ss << "GatherDictionaryArgs has no target dictionary name; ";

  if (!message.has_data_path() && (message.batch_path_size() == 0))
    ss << "GatherDictionaryArgs has neither batch_path nor data_path set; ";

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::DictionaryData& message) {
  std::stringstream ss;

  if (!message.has_name())
    ss << "DictionaryData has no dictionary name; ";

  bool is_token_df_ok = message.token_df_size() == 0 || message.token_df_size() == message.token_size();
  bool is_token_tf_ok = message.token_tf_size() == 0 || message.token_tf_size() == message.token_size();
  bool is_token_value_ok = message.token_value_size() == 0 || message.token_value_size() == message.token_size();

  if (message.token_size() != message.class_id_size() ||
      !is_token_df_ok ||
      !is_token_tf_ok ||
      !is_token_value_ok) {
    ss << "DictionaryData general token fields have inconsistent sizes; ";
  }

  if (message.cooc_first_index_size() != message.cooc_second_index_size() ||
      message.cooc_first_index_size() != message.cooc_value_size()) {
    ss << "DictionaryData cooc fields have inconsistent sizes; ";
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::ExportModelArgs& message) {
  std::stringstream ss;
  if (!message.has_file_name()) ss << "ExportModelArgs.file_name is not defined; ";
  if (!message.has_model_name()) ss << "ExportModelArgs.model_name is not defined; ";

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::ImportModelArgs& message) {
  std::stringstream ss;
  if (!message.has_file_name()) ss << "ImportModelArgs.file_name is not defined; ";
  if (!message.has_model_name()) ss << "ImportModelArgs.model_name is not defined; ";

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::ImportDictionaryArgs& message) {
  std::stringstream ss;
  if (!message.has_file_name()) ss << "ImportDictionaryArgs.file_name is not defined; ";
  if (!message.has_dictionary_name())
    ss << "ImportDictionaryArgs.dictionary_name is not defined; ";

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::ProcessBatchesArgs& message) {
  std::stringstream ss;

  if (message.batch_filename_size() == 0 && message.batch_size() == 0)
    ss << "Either ProcessBatchesArgs.batch_filename or ProcessBatchesArgs.batch must be specified; ";
  if (message.batch_filename_size() != 0 && message.batch_size() != 0)
    ss << "Only one of ProcessBatchesArgs.batch_filename, "
       << "ProcessBatchesArgs.batch must be specified; ";

  if (message.batch_filename_size() != 0 && message.batch_filename_size() != message.batch_weight_size())
    ss << "Length mismatch in fields ProcessBatchesArgs.batch_filename and ProcessBatchesArgs.batch_weight";

  if (message.batch_size() != 0 && message.batch_size() != message.batch_weight_size())
    ss << "Length mismatch in fields ProcessBatchesArgs.batch_filename and ProcessBatchesArgs.batch_weight";

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::ImportBatchesArgs& message) {
  std::stringstream ss;

  if (message.batch_name_size() != 0 && message.batch_name_size() != message.batch_size())
    ss << "Length mismatch in fields ImportBatchesArgs.batch_name and ImportBatchesArgs.batch";

  return ss.str();
}

// Empty ValidateMessage routines
inline std::string DescribeErrors(const ::artm::GetTopicModelArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::RegularizerInternalState& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::InvokeIterationArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::MergeModelArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::RegularizeModelArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::NormalizeModelArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::RegularizerConfig& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::SynchronizeModelArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::ExportDictionaryArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::ScoreData& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::MasterComponentInfo& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::GetDictionaryArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::GetMasterComponentInfoArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::GetRegularizerStateArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::ProcessBatchesResult& message) { return std::string(); }

///////////////////////////////////////////////////////////////////////////////////////////////////
// FixMessage routines (optional)
///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline void FixMessage(T* message) {}

template<>
inline void FixMessage(::artm::TopicModel* message) {
  const int token_size = message->token_size();
  if ((message->class_id_size() == 0) && (token_size > 0)) {
    message->mutable_class_id()->Reserve(token_size);
    for (int i = 0; i < token_size; ++i)
      message->add_class_id(::artm::core::DefaultClass);
  }

  if (message->topic_name_size() > 0)
    message->set_topics_count(message->topic_name_size());
}

template<>
inline void FixMessage(::artm::ModelConfig* message) {
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

  if (message->regularizer_settings_size() == 0) {
    // using old version of parameters, convert to new one
    if (message->regularizer_tau_size() == 0) {
      for (int i = 0; i < message->regularizer_name_size(); ++i)
        message->add_regularizer_tau(1.0);
    }

    for (int i = 0; i < message->regularizer_name_size(); ++i) {
      auto settings = message->add_regularizer_settings();
      settings->set_name(message->regularizer_name(i));
      settings->set_use_relative_regularization(false);
      settings->set_tau(message->regularizer_tau(i));
    }
  } else {
    // using new version of parameters, skip old one
    for (int i = 0; i < message->regularizer_settings_size(); ++i) {
      if (!message->regularizer_settings(i).has_tau())
        message->mutable_regularizer_settings(i)->set_tau(1.0);

      if (!message->regularizer_settings(i).has_use_relative_regularization())
        message->mutable_regularizer_settings(i)->set_use_relative_regularization(false);

      if (message->regularizer_settings(i).use_relative_regularization() &&
        !message->regularizer_settings(i).has_gamma())
        message->mutable_regularizer_settings(i)->set_gamma(1.0);
    }
  }
}

template<>
inline void FixMessage(::artm::Batch* message) {
  if (message->class_id_size() == 0) {
    for (int i = 0; i < message->token_size(); ++i) {
      message->add_class_id(DefaultClass);
    }
  }

  // Upgrade token_count to token_weight
  for (::artm::Item& item : *message->mutable_item()) {
    for (::artm::Field& field : *item.mutable_field()) {
      if (field.token_count_size() != 0 && field.token_weight_size() == 0) {
        field.mutable_token_weight()->Reserve(field.token_count_size());
        for (int i = 0; i < field.token_count_size(); ++i)
          field.add_token_weight(static_cast<float>(field.token_count(i)));
        field.clear_token_count();
      }
    }
  }
}

template<>
inline void FixMessage(::artm::GetThetaMatrixArgs* message) {
  if (message->has_batch())
    FixMessage(message->mutable_batch());

  if (message->has_use_sparse_format())
    message->set_matrix_layout(GetThetaMatrixArgs_MatrixLayout_Sparse);
}

template<>
inline void FixMessage(::artm::GetTopicModelArgs* message) {
  if (message->has_use_sparse_format())
    message->set_matrix_layout(GetTopicModelArgs_MatrixLayout_Sparse);
}

template<>
inline void FixMessage(::artm::GetScoreValueArgs* message) {
  if (message->has_batch()) FixMessage(message->mutable_batch());
}

template<>
inline void FixMessage(::artm::InitializeModelArgs* message) {
  if (message->topic_name_size() == 0) {
    for (int i = 0; i < message->topics_count(); ++i) {
      message->add_topic_name("@topic_" + std::to_string(i));
    }
  } else {
    message->set_topics_count(message->topic_name_size());
  }
}

template<>
inline void FixMessage(::artm::DictionaryData* message) {
  if (message->class_id_size() == 0) {
    for (int i = 0; i < message->token_size(); ++i) {
      message->add_class_id(DefaultClass);
    }
  }
}

template<>
inline void FixMessage(::artm::ProcessBatchesArgs* message) {
  if (message->batch_weight_size() == 0) {
    int size = message->batch_filename_size() > 0 ? message->batch_filename_size() : message->batch_size();
    for (int i = 0; i < size; ++i)
      message->add_batch_weight(1.0f);
  }

  for (int i = 0; i < message->batch_size(); ++i)
    FixMessage(message->mutable_batch(i));
}

template<>
inline void FixMessage(::artm::MasterModelConfig* message) {
  if (message->class_weight_size() == 0) {
    for (int i = 0; i < message->class_id_size(); ++i)
      message->add_class_weight(1.0f);
  }
}

template<>
inline void FixMessage(::artm::FitOfflineMasterModelArgs* message) {
  if (message->batch_weight_size() == 0) {
    for (int i = 0; i < message->batch_filename_size(); ++i)
      message->add_batch_weight(1.0f);
  }
}

template<>
inline void FixMessage(::artm::FitOnlineMasterModelArgs* message) {
  if (message->batch_weight_size() == 0) {
    for (int i = 0; i < message->batch_filename_size(); ++i)
      message->add_batch_weight(1.0f);
  }

  if (message->apply_weight_size() == 0) {
    for (int i = 0; i < message->decay_weight_size(); ++i)
      message->add_apply_weight(1.0f - message->decay_weight(i));
  }

  if (message->decay_weight_size() == 0) {
    for (int i = 0; i < message->apply_weight_size(); ++i) {
      message->add_decay_weight(1.0f - message->apply_weight(i));
    }
  }
}

template<>
inline void FixMessage(::artm::TransformMasterModelArgs* message) {
  for (int i = 0; i < message->batch_size(); ++i)
    FixMessage(message->mutable_batch(i));
}

template<>
inline void FixMessage(::artm::ImportBatchesArgs* message) {
  for (int i = 0; i < message->batch_size(); ++i)
    FixMessage(message->mutable_batch(i));

  if (message->batch_name_size() == 0) {
    for (int i = 0; i < message->batch_size(); ++i)
      message->add_batch_name(message->batch(i).id());
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// DescribeMessage routines (optional)
///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline std::string DescribeMessage(const T& message) { return std::string(); }

template<>
inline std::string DescribeMessage(const ::artm::RegularizerSettings& message) {
  std::stringstream ss;
  ss << ", regularizer=(name:" << message.name() <<
        ", tau:" << message.tau();
  if (message.use_relative_regularization())
    ss << "relative_regularization:True, gamma:" << message.gamma() << ")";
  else
    ss << "relative_regularization:False" << ")";
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::ModelConfig& message) {
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
  for (int i = 0; i < message.regularizer_settings_size(); ++i)
    DescribeMessage(message.regularizer_settings(i));
  for (int i = 0; i < message.class_id_size(); ++i)
    ss << ", class=(" << message.class_id(i) << ":" << message.class_weight(i) << ")";
  ss << ", use_sparse_bow=" << (message.use_sparse_bow() ? "yes" : "no");
  ss << ", use_random_theta=" << (message.use_random_theta() ? "yes" : "no");
  ss << ", use_new_tokens=" << (message.use_new_tokens() ? "yes" : "no");
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::MasterComponentConfig& message) {
  std::stringstream ss;
  ss << "MasterComponentConfig";
  ss << ": disk_path=" << message.disk_path();
  ss << ", stream_size=" << message.stream_size();
  ss << ", compact_batches=" << (message.compact_batches() ? "yes" : "no");
  ss << ", cache_theta=" << (message.cache_theta() ? "yes" : "no");
  ss << ", processors_count=" << message.processors_count();
  ss << ", processor_queue_max_size=" << message.processor_queue_max_size();
  ss << ", merger_queue_max_size=" << message.merger_queue_max_size();
  ss << ", score_config_size=" << message.score_config_size();
  ss << ", disk_cache_path" << message.disk_cache_path();
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::InitializeModelArgs& message) {
  std::stringstream ss;
  ss << "InitializeModelArgs";
  ss << ": model_name=" << message.model_name();

  if (message.has_dictionary_name())
    ss << ", dictionary_name=" << message.dictionary_name();
  if (message.has_topics_count())
    ss << ", topics_count=" << message.topics_count();
  ss << ", topic_name_size=" << message.topic_name_size();
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::FilterDictionaryArgs& message) {
  std::stringstream ss;
  ss << "FilterDictionaryArgs";
  ss << ": dictionary_name=" << message.dictionary_name();

  if (message.has_class_id())
    ss << ", class_id=" << message.class_id();
  if (message.has_min_df())
    ss << ", min_df=" << message.min_df();
  if (message.has_max_df())
    ss << ", max_df=" << message.max_df();
  if (message.has_min_tf())
    ss << ", min_tf=" << message.min_tf();
  if (message.has_max_tf())
    ss << ", max_tf=" << message.max_tf();

  if (message.has_min_df_rate())
    ss << ", min_df_rate=" << message.min_df_rate();
  if (message.has_max_df_rate())
    ss << ", max_df_rate=" << message.max_df_rate();

  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::GatherDictionaryArgs& message) {
  std::stringstream ss;
  ss << "GatherDictionaryArgs";
  ss << ": dictionary_target_name=" << message.dictionary_target_name();

  if (message.has_data_path())
    ss << ", data_path=" << message.data_path();
  if (message.has_cooc_file_path())
    ss << ", cooc_file_path=" << message.cooc_file_path();
  if (message.has_vocab_file_path())
    ss << ", vocab_file_path=" << message.vocab_file_path();
  ss << ", symmetric_cooc_values=" << message.symmetric_cooc_values();

  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::ProcessBatchesArgs& message) {
  std::stringstream ss;
  ss << "ProcessBatchesArgs";
  ss << ": nwt_target_name=" << message.nwt_target_name();
  ss << ", batch_filename_size=" << message.batch_filename_size();
  ss << ", batch_size=" << message.batch_size();
  ss << ", batch_weight_size=" << message.batch_weight_size();
  ss << ", pwt_source_name=" << message.pwt_source_name();
  ss << ", inner_iterations_count=" << message.inner_iterations_count();
  ss << ", stream_name=" << message.stream_name();
  for (int i = 0; i < message.regularizer_name_size(); ++i)
    ss << ", regularizer=(name:" << message.regularizer_name(i) << ", tau:" << message.regularizer_tau(i) << ")";
  for (int i = 0; i < message.class_id_size(); ++i)
    ss << ", class=(" << message.class_id(i) << ":" << message.class_weight(i) << ")";
  ss << ", reuse_theta=" << (message.reuse_theta() ? "yes" : "no");
  ss << ", opt_for_avx=" << (message.opt_for_avx() ? "yes" : "no");
  ss << ", use_sparse_bow=" << (message.use_sparse_bow() ? "yes" : "no");
  ss << ", reset_scores=" << (message.reset_scores() ? "yes" : "no");
  ss << ", predict_class_id=" << (message.predict_class_id());
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::NormalizeModelArgs& message) {
  std::stringstream ss;
  ss << "NormalizeModelArgs";
  ss << ": pwt_target_name=" << message.pwt_target_name();
  ss << ", nwt_source_name=" << message.nwt_source_name();
  ss << ", rwt_source_name=" << message.rwt_source_name();
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::MergeModelArgs& message) {
  std::stringstream ss;
  ss << "MergeModelArgs";
  ss << ": nwt_target_name=" << message.nwt_target_name();
  for (int i = 0; i < message.nwt_source_name_size(); ++i)
    ss << ", class=(" << message.nwt_source_name(i) << ":" << message.source_weight(i) << ")";
  ss << ", topic_name_size=" << message.topic_name_size();
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::RegularizeModelArgs& message) {
  std::stringstream ss;
  ss << "RegularizeModelArgs";
  ss << ": rwt_target_name=" << message.rwt_target_name();
  ss << ", pwt_source_name=" << message.pwt_source_name();
  ss << ", nwt_source_name=" << message.nwt_source_name();
  for (int i = 0; i < message.regularizer_settings_size(); ++i)
    DescribeMessage(message.regularizer_settings(i));
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::MasterModelConfig& message) {
  std::stringstream ss;
  ss << "MasterModelConfig";
  ss << ": topic_name_size=" << message.topic_name_size();
  for (int i = 0; i < message.class_id_size(); ++i)
    ss << ", class=(" << message.class_id(i) << ":" << message.class_weight(i) << ")";
  ss << ", score_config_size=" << message.score_config_size();
  ss << ", threads=" << message.threads();
  ss << ", pwt_name=" << message.pwt_name();
  ss << ", nwt_name=" << message.nwt_name();
  ss << ", inner_iterations_count=" << message.inner_iterations_count();
  for (int i = 0; i < message.regularizer_config_size(); ++i)
    ss << ", regularizer=("
       << message.regularizer_config(i).name() << ":"
       << message.regularizer_config(i).tau() << ")";
  ss << ", reuse_theta=" << message.reuse_theta() ? "yes" : "no";
  ss << ", opt_for_avx=" << message.opt_for_avx() ? "yes" : "no";
  ss << ", use_sparse_bow=" << message.use_sparse_bow() ? "yes" : "no";
  ss << ", disk_cache_path" << message.disk_cache_path();

  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::FitOfflineMasterModelArgs& message) {
  std::stringstream ss;
  ss << "FitOfflineMasterModelArgs";
  ss << ", batch_filename_size=" << message.batch_filename_size();
  ss << ", batch_weight_size=" << message.batch_weight_size();
  ss << ", passes=" << message.passes();
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::FitOnlineMasterModelArgs& message) {
  std::stringstream ss;
  ss << "FitOnlineMasterModelArgs";
  ss << ", batch_filename_size=" << message.batch_filename_size();
  ss << ", batch_weight_size=" << message.batch_weight_size();
  ss << ", update_after:apply_weight:decay_weight=(";
  for (int i = 0; i < message.update_after_size(); ++i) {
    if (i != 0) ss << ", ";
    ss << message.update_after(i) << ":";
    ss << message.apply_weight(i) << ":";
    ss << message.decay_weight(i);
  }
  ss << ")";
  ss << ", async=" << message.async() ? "yes" : "no";
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::TransformMasterModelArgs& message) {
  std::stringstream ss;
  ss << "TransformMasterModelArgs";
  ss << ", batch_filename_size=" << message.batch_filename_size();
  ss << ", batch_size=" << message.batch_size();
  ss << ", theta_matrix_type=" << message.theta_matrix_type();
  ss << ", predict_class_id=" << message.predict_class_id();
  return ss.str();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Templates
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline bool ValidateMessage(const T& message, bool throw_error) {
  std::string ss = DescribeErrors(message);

  if (ss.empty())
    return true;

  if (throw_error)
    BOOST_THROW_EXCEPTION(InvalidOperation(ss));
  LOG(WARNING) << ss;
  return false;
}

template <typename T>
inline bool FixAndValidateMessage(T* message, bool throw_error) {
  FixMessage(message);
  return ValidateMessage(*message, throw_error);
}

}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_CHECK_MESSAGES_H_
