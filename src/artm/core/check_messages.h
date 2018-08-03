// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <set>
#include <string>
#include <vector>
#include<unordered_set>

#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"

#include "glog/logging.h"

#include "artm/core/common.h"
#include "artm/core/exceptions.h"
#include "artm/core/token.h"
#include "artm/core/transaction_type.h"
#include "artm/core/protobuf_serialization.h"

namespace artm {
namespace core {

template <typename T>
inline bool FixAndValidateMessage(T* message, bool throw_error = true);

template <typename T>
inline bool ValidateMessage(const T& message, bool throw_error = true);

template<typename T>
inline void FixPackedMessage(std::string* message);

///////////////////////////////////////////////////////////////////////////////////////////////////
// DescribeErrors routines
// This method is required for all messages that go through c_interface.
///////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string DescribeErrors(const ::artm::TopicModel& message) {
  std::stringstream ss;

  const bool has_topic_data = (message.num_topics() != 0 || message.topic_name_size() != 0);
  const bool has_token_data = (message.class_id_size() != 0 || message.token_size() != 0);
  const bool has_bulk_data = (message.token_weights_size() != 0);
  const bool has_sparse_format = has_bulk_data && (message.topic_indices_size() != 0);

  if (has_topic_data) {
    if (message.num_topics() != message.topic_name_size()) {
      ss << "Length mismatch in fields TopicModel.num_topics and TopicModel.topic_name";
    }
  }

  if (has_token_data) {
    if (message.class_id_size() != message.token_size()) {
      ss << "Inconsistent fields size in TopicModel.token and TopicModel.class_id: "
         << message.token_size() << " vs " << message.class_id_size();
    }
  }

  if (has_bulk_data && !has_topic_data) {
    ss << "TopicModel.topic_name_size is empty";
  }
  if (has_bulk_data && !has_token_data) {
    ss << "TopicModel.token_size is empty";
  }

  if (has_bulk_data) {
    if ((message.token_weights_size() != message.token_size()) ||
        (has_sparse_format && (message.topic_indices_size() != message.token_size()))) {
      ss << "Inconsistent fields size in TopicModel: "
        << message.token_size() << " vs " << message.class_id_size()
        << " vs " << message.token_weights_size() << ";";
    }

    for (int i = 0; i < message.token_size(); ++i) {
      bool has_sparse_format_local = has_sparse_format && (message.topic_indices(i).value_size() > 0);
      if (has_sparse_format_local) {
        if (message.topic_indices(i).value_size() != message.token_weights(i).value_size()) {
          ss << "Length mismatch between TopicModel.topic_indices(" << i << ")"
             << " and TopicModel.token_weights(" << i << ")";
          break;
        }

        bool ok = true;
        for (int topic_indices : message.topic_indices(i).value()) {
          if (topic_indices < 0 || topic_indices >= message.num_topics()) {
            ss << "Value " << topic_indices << " in message.topic_indices(" << i
               << ") is negative or exceeds TopicModel.num_topics";
            ok = false;
            break;
          }
        }

        if (!ok) {
          break;
        }
      }

      if (!has_sparse_format) {
        if (message.token_weights(i).value_size() != message.num_topics()) {
          ss << "Length mismatch between TopicModel.num_topics and TopicModel.token_weights(" << i << ")";
          break;
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
  const bool has_sparse_format = (message.topic_indices_size() != 0);
  if ((message.item_weights_size() != item_size) ||
      (has_title && (message.item_title_size() != item_size)) ||
      (has_sparse_format && (message.topic_indices_size() != item_size))) {
    ss << "Inconsistent fields size in ThetaMatrix: "
       << message.item_id_size() << " vs " << message.item_weights_size()
       << " vs " << message.item_title_size() << " vs " << message.topic_indices_size() << ";";
  }

  if (message.num_topics() == 0 || message.topic_name_size() == 0) {
    ss << "ThetaMatrix.topic_name_size is empty";
  }
  if (message.num_topics() != message.topic_name_size()) {
    ss << "Length mismatch in fields ThetaMatrix.num_topics and ThetaMatrix.topic_name";
  }

  for (int i = 0; i < message.item_id_size(); ++i) {
    if (has_sparse_format) {
      if (message.topic_indices(i).value_size() != message.item_weights(i).value_size()) {
        ss << "Length mismatch between ThetaMatrix.topic_indices(" << i << ")"
           << " and ThetaMatrix.item_weights(" << i << ")";
        break;
      }

      bool ok = true;
      for (int topic_indices : message.topic_indices(i).value()) {
        if (topic_indices < 0 || topic_indices >= message.num_topics()) {
          ss << "Value " << topic_indices << " in message.topic_indices(" << i
             << ") is negative or exceeds ThetaMatrix.num_topics";
          ok = false;
          break;
        }
      }

      if (!ok) {
        break;
      }
    }
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::Item& message) {
  std::stringstream ss;
  std::string id = "NO_ID";
  id = message.has_id() ? std::to_string(message.id()) : id;
  id = message.has_title() ? message.title() : id;

  if (message.token_id_size() == 0 && message.token_weight_size() > 0) {
    ss << "Item " << id << " has empty token_id with non-empty token_weight\n";
  }

  if (message.transaction_start_index_size() != message.transaction_typename_id_size() + 1) {
    ss << "Item " << id << " has incocnsistent transaction_start_index_size ("
       << message.transaction_start_index_size() << ") and transaction_typename_id_size + 1 ("
       << message.transaction_typename_id_size() + 1 << ")\n";
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
  if (!has_tokens) {
    ss << "Empty Batch.token is no longer supported, batch.id = " << message.id();
    return ss.str();
  }

  if (has_tokens && (message.class_id_size() != message.token_size())) {
    ss << "Length mismatch in fields Batch.class_id and Batch.token, batch.id = " << message.id();
    return ss.str();
  }

  for (int item_id = 0; item_id < message.item_size(); ++item_id) {
    ss << DescribeErrors(message.item(item_id));
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::GetScoreValueArgs& message) {
  std::stringstream ss;

  if (!message.has_score_name() || message.score_name().empty()) {
    ss << "GetScoreValueArgs.score_name is missing; ";
  }
  return ss.str();
}

inline std::string DescribeErrors(const ::artm::MasterModelConfig& message) {
  std::stringstream ss;

  if (message.class_weight_size() != message.class_id_size()) {
    ss << "Length mismatch in fields MasterModelConfig.class_id and MasterModelConfig.class_weight; ";
  }

  if (message.num_document_passes() < 0) {
    ss << "Field MasterModelConfig.num_document_passes must be non-negative; ";
  }

  for (int i = 0; i < message.regularizer_config_size(); ++i) {
    const RegularizerConfig& config = message.regularizer_config(i);
    if (!config.has_tau()) {
      ss << "Field MasterModelConfig.RegularizerConfig.tau must not be empty "
         << "(regularizer name: " << config.name() << "); ";
    }
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::FitOfflineMasterModelArgs& message) {
  std::stringstream ss;

  if (message.batch_filename_size() != message.batch_weight_size()) {
    ss << "Length mismatch in fields FitOfflineMasterModelArgs.batch_filename "
       << "and FitOfflineMasterModelArgs.batch_weight; ";
  }

  if (message.num_collection_passes() <= 0) {
    ss << "FitOfflineMasterModelArgs.passes() must be a positive number";
  }

  if (message.has_batch_folder() && (message.batch_filename_size() != 0)) {
    ss << "Only one of FitOfflineMasterModelArgs.batch_folder, "
       << "FitOfflineMasterModelArgs.batch_filename must be specified; ";
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::FitOnlineMasterModelArgs& message) {
  std::stringstream ss;

  if (message.batch_filename_size() == 0) {
    ss << "Fields FitOnlineMasterModelArgs.batch_filename must not be empty; ";
  }

  if (message.batch_filename_size() != message.batch_weight_size()) {
    ss << "Length mismatch in fields FitOnlineMasterModelArgs.batch_filename "
    << "and FitOnlineMasterModelArgs.batch_weight; ";
  }

  if (message.update_after_size() == 0) {
    ss << "Field FitOnlineMasterModelArgs.update_after must not be empty; ";
  }

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

  if (message.batch_filename_size() == 0 && message.batch_size() == 0) {
    ss << "Either TransformMasterModelArgs.batch_filename or TransformMasterModelArgs.batch must be specified; ";
  }
  if (message.batch_filename_size() != 0 && message.batch_size() != 0) {
    ss << "Only one of TransformMasterModelArgs.batch_filename, "
       << "TransformMasterModelArgs.batch must be specified; ";
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::InitializeModelArgs& message) {
  std::stringstream ss;

  if (!message.has_model_name()) {
    // Allow this to default to MasterModelConfig.pwt_name
    // ss << "InitializeModelArgs.model_name is not defined; ";
  }

  if (!message.has_dictionary_name()) {
    // Allow this to initialize an existing model
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::FilterDictionaryArgs& message) {
  std::stringstream ss;

  if (!message.has_dictionary_name()) {
    ss << "FilterDictionaryArgs has no dictionary name; ";
  }

  if (!message.has_dictionary_target_name()) {
     ss << "FilterDictionaryArgs has no target dictionary name; ";
  }

  if (message.has_max_dictionary_size() && (message.max_dictionary_size() <= 0)) {
    ss << "FilterDictionaryArgs.max_dictionary_size must be positive integer; ";
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::GatherDictionaryArgs& message) {
  std::stringstream ss;

  if (!message.has_dictionary_target_name()) {
    ss << "GatherDictionaryArgs has no target dictionary name; ";
  }

  if (!message.has_data_path() && (message.batch_path_size() == 0)) {
    ss << "GatherDictionaryArgs has neither batch_path nor data_path set; ";
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::DictionaryData& message) {
  std::stringstream ss;

  if (!message.has_name()) {
    ss << "DictionaryData has no dictionary name; ";
  }

  bool is_token_df_ok = message.token_df_size() == 0 || message.token_df_size() == message.token_size();
  bool is_token_tf_ok = message.token_tf_size() == 0 || message.token_tf_size() == message.token_size();
  bool is_token_value_ok = message.token_value_size() == 0 || message.token_value_size() == message.token_size();

  if (message.token_size() != message.class_id_size() ||
      !is_token_df_ok ||
      !is_token_tf_ok ||
      !is_token_value_ok) {
    ss << "DictionaryData general token fields have inconsistent sizes; ";
  }

  bool fst_size = message.cooc_first_index_size();
  bool snd_size = message.cooc_second_index_size();
  bool val_size = message.cooc_value_size();
  bool tf_size = message.cooc_tf_size();
  bool df_size = message.cooc_df_size();

  if ((fst_size != snd_size) ||
      (fst_size != val_size) ||
      (tf_size != df_size) ||
      (tf_size > 0 && tf_size != fst_size)) {
    ss << "DictionaryData cooc fields have inconsistent sizes; ";
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::ExportModelArgs& message) {
  std::stringstream ss;
  if (!message.has_file_name()) {
    ss << "ExportModelArgs.file_name is not defined; ";
  }

  // Allow this to default to MasterModelConfig.pwt_name
  // if (!message.has_model_name()) ss << "ExportModelArgs.model_name is not defined; ";

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::ImportModelArgs& message) {
  std::stringstream ss;
  if (!message.has_file_name()) {
    ss << "ImportModelArgs.file_name is not defined; ";
  }

  // Allow this to default to MasterModelConfig.pwt_name
  // if (!message.has_model_name()) ss << "ImportModelArgs.model_name is not defined; ";

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::ExportScoreTrackerArgs& message) {
  std::stringstream ss;
  if (!message.has_file_name()) {
    ss << "ExportScoreTrackerArgs.file_name is not defined; ";
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::ImportScoreTrackerArgs& message) {
  std::stringstream ss;
  if (!message.has_file_name()) {
    ss << "ImportScoreTrackerArgs.file_name is not defined; ";
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::ImportDictionaryArgs& message) {
  std::stringstream ss;
  if (!message.has_file_name()) {
    ss << "ImportDictionaryArgs.file_name is not defined; ";
  }

  if (!message.has_dictionary_name()) {
    ss << "ImportDictionaryArgs.dictionary_name is not defined; ";
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::ProcessBatchesArgs& message) {
  std::stringstream ss;

  if (message.batch_filename_size() == 0 && message.batch_size() == 0) {
    ss << "Either ProcessBatchesArgs.batch_filename or ProcessBatchesArgs.batch must be specified; ";
  }
  if (message.batch_filename_size() != 0 && message.batch_size() != 0) {
    ss << "Only one of ProcessBatchesArgs.batch_filename, "
       << "ProcessBatchesArgs.batch must be specified; ";
  }

  if (message.batch_filename_size() != 0 && message.batch_filename_size() != message.batch_weight_size()) {
    ss << "Length mismatch in fields ProcessBatchesArgs.batch_filename and ProcessBatchesArgs.batch_weight";
  }

  if (message.batch_size() != 0 && message.batch_size() != message.batch_weight_size()) {
    ss << "Length mismatch in fields ProcessBatchesArgs.batch_filename and ProcessBatchesArgs.batch_weight";
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::ImportBatchesArgs& message) {
  std::stringstream ss;

  if (message.batch_size() == 0) {
    ss << "Empty ImportBatchesArgs.batch field";
  }

  return ss.str();
}

inline std::string DescribeErrors(const ::artm::MergeModelArgs& message) {
  std::stringstream ss;

  if (message.source_weight_size() != 0 && message.source_weight_size() != message.nwt_source_name_size()) {
    ss << "Length mismatch in fields MergeModelArgs.source_weight and MergeModelArgs.nwt_source_name";
  }

  return ss.str();
}

// Empty ValidateMessage routines
inline std::string DescribeErrors(const ::artm::GetTopicModelArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::GetThetaMatrixArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::RegularizeModelArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::NormalizeModelArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::RegularizerConfig& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::ExportDictionaryArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::ScoreData& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::MasterComponentInfo& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::GetDictionaryArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::GetMasterComponentInfoArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::ProcessBatchesResult& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::ClearThetaCacheArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::ClearScoreCacheArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::ClearScoreArrayCacheArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::ScoreArray& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::GetScoreArrayArgs& message) { return std::string(); }
inline std::string DescribeErrors(const ::artm::CollectionParserConfig& message) { return std::string(); }

///////////////////////////////////////////////////////////////////////////////////////////////////
// FixMessage routines (optional)
///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline void FixMessage(T* message) { }

#define FIX_REGULARIZER_CONFIG(T, U) if (message->type() == T) { message->set_config(ProtobufSerialization::ConvertJsonToBinary< U>(message->config_json())); fixed = true; }  // NOLINT
#define FIX_SCORE_CONFIG(T, U) if (message->type() == T) { message->set_config(ProtobufSerialization::ConvertJsonToBinary< U>(message->config_json())); fixed = true; }  // NOLINT
#define FIX_SCORE_DATA(T, U) if (message->type() == T) { message->set_data_json( ProtobufSerialization::ConvertBinaryToJson< U>(message->data())); fixed = true; }  // NOLINT

template<>
inline void FixMessage(::artm::RegularizerConfig* message) {
  if (ProtobufSerialization::singleton().IsJson() && message->has_config_json() && !message->has_config()) {
    bool fixed = false;
    FIX_REGULARIZER_CONFIG(RegularizerType_SmoothSparseTheta, ::artm::SmoothSparseThetaConfig);
    FIX_REGULARIZER_CONFIG(RegularizerType_SmoothSparsePhi, ::artm::SmoothSparsePhiConfig);
    FIX_REGULARIZER_CONFIG(RegularizerType_DecorrelatorPhi, ::artm::DecorrelatorPhiConfig);
    FIX_REGULARIZER_CONFIG(RegularizerType_MultiLanguagePhi, ::artm::MultiLanguagePhiConfig);
    FIX_REGULARIZER_CONFIG(RegularizerType_LabelRegularizationPhi, ::artm::LabelRegularizationPhiConfig);
    FIX_REGULARIZER_CONFIG(RegularizerType_SpecifiedSparsePhi, ::artm::SpecifiedSparsePhiConfig);
    FIX_REGULARIZER_CONFIG(RegularizerType_ImproveCoherencePhi, ::artm::ImproveCoherencePhiConfig);
    FIX_REGULARIZER_CONFIG(RegularizerType_SmoothPtdw, ::artm::SmoothPtdwConfig);
    FIX_REGULARIZER_CONFIG(RegularizerType_TopicSelectionTheta, ::artm::TopicSelectionThetaConfig);
    FIX_REGULARIZER_CONFIG(RegularizerType_BitermsPhi, ::artm::BitermsPhiConfig);
    FIX_REGULARIZER_CONFIG(RegularizerType_HierarchySparsingTheta, ::artm::HierarchySparsingThetaConfig);
    if (!fixed) {
      BOOST_THROW_EXCEPTION(InternalError("Given RegularizerType is not supported for json serialization"));
    }
  }

  if (message->has_gamma() && (message->gamma() < 0) || (message->gamma() > 1)) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
      "Regularization parameter 'gamma' must be between 0 and 1. "
      "Refer to documentation for more details. "));
  }
}

template<>
inline void FixMessage(::artm::ScoreConfig* message) {
  if (ProtobufSerialization::singleton().IsJson() && message->has_config_json() && !message->has_config()) {
    bool fixed = false;
    FIX_SCORE_CONFIG(ScoreType_Perplexity, ::artm::PerplexityScoreConfig);
    FIX_SCORE_CONFIG(ScoreType_SparsityTheta, ::artm::SparsityThetaScoreConfig);
    FIX_SCORE_CONFIG(ScoreType_SparsityPhi, ::artm::SparsityPhiScoreConfig);
    FIX_SCORE_CONFIG(ScoreType_ItemsProcessed, ::artm::ItemsProcessedScoreConfig);
    FIX_SCORE_CONFIG(ScoreType_TopTokens, ::artm::TopTokensScoreConfig);
    FIX_SCORE_CONFIG(ScoreType_ThetaSnippet, ::artm::ThetaSnippetScoreConfig);
    FIX_SCORE_CONFIG(ScoreType_TopicKernel, ::artm::TopicKernelScoreConfig);
    FIX_SCORE_CONFIG(ScoreType_TopicMassPhi, ::artm::TopicMassPhiScoreConfig);
    FIX_SCORE_CONFIG(ScoreType_ClassPrecision, ::artm::ClassPrecisionScoreConfig);
    FIX_SCORE_CONFIG(ScoreType_PeakMemory, ::artm::PeakMemoryScoreConfig);
    FIX_SCORE_CONFIG(ScoreType_BackgroundTokensRatio, ::artm::BackgroundTokensRatioScoreConfig);
    if (!fixed) {
      BOOST_THROW_EXCEPTION(InternalError("Given ScoreType is not supported for json serialization"));
    }
  }

  if (message->type() == ScoreType_TopTokens) {
    FixPackedMessage<TopTokensScoreConfig>(message->mutable_config());
  }
}

template<>
inline void FixMessage(::artm::ScoreData* message) {
  if (ProtobufSerialization::singleton().IsJson() && message->has_data() && !message->has_data_json()) {
    bool fixed = false;
    FIX_SCORE_DATA(ScoreType_Perplexity, ::artm::PerplexityScore);
    FIX_SCORE_DATA(ScoreType_SparsityTheta, ::artm::SparsityThetaScore);
    FIX_SCORE_DATA(ScoreType_SparsityPhi, ::artm::SparsityPhiScore);
    FIX_SCORE_DATA(ScoreType_ItemsProcessed, ::artm::ItemsProcessedScore);
    FIX_SCORE_DATA(ScoreType_TopTokens, ::artm::TopTokensScore);
    FIX_SCORE_DATA(ScoreType_ThetaSnippet, ::artm::ThetaSnippetScore);
    FIX_SCORE_DATA(ScoreType_TopicKernel, ::artm::TopicKernelScore);
    FIX_SCORE_DATA(ScoreType_TopicMassPhi, ::artm::TopicMassPhiScore);
    FIX_SCORE_DATA(ScoreType_ClassPrecision, ::artm::ClassPrecisionScore);
    FIX_SCORE_DATA(ScoreType_PeakMemory, ::artm::PeakMemoryScore);
    FIX_SCORE_DATA(ScoreType_BackgroundTokensRatio, ::artm::BackgroundTokensRatioScore);
    if (!fixed) {
      BOOST_THROW_EXCEPTION(InternalError("Given ScoreType is not supported for json de-serialization"));
    }
  }
}

#undef FIX_REGULARIZER_CONFIG
#undef FIX_SCORE_CONFIG
#undef FIX_SCORE_DATA

template<>
inline void FixMessage(::artm::TopicModel* message) {
  const int token_size = message->token_size();
  if ((message->class_id_size() == 0) && (token_size > 0)) {
    message->mutable_class_id()->Reserve(token_size);
    for (int i = 0; i < token_size; ++i) {
      message->add_class_id(::artm::core::DefaultClass);
    }
  }

  if (message->topic_name_size() > 0) {
    message->set_num_topics(message->topic_name_size());
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
  for (auto& item : *message->mutable_item()) {
    for (auto& field : *item.mutable_field()) {
      if (field.token_count_size() != 0 && field.token_weight_size() == 0) {
        field.mutable_token_weight()->Reserve(field.token_count_size());
        for (int i = 0; i < field.token_count_size(); ++i) {
          field.add_token_weight(static_cast<float>(field.token_count(i)));
        }
        field.clear_token_count();
      }
    }
  }

  // Upgrade away from Field
  for (auto& item : *message->mutable_item()) {
    for (auto& field : *item.mutable_field()) {
      item.mutable_token_id()->MergeFrom(field.token_id());
      item.mutable_token_weight()->MergeFrom(field.token_weight());
    }

    item.clear_field();
  }

  // For items without title set title to item id
  for (auto& item : *message->mutable_item()) {
    if (!item.has_title() && item.has_id()) {
      item.set_title(boost::lexical_cast<std::string>(item.id()));
    }
  }

  // old-style batch should be filled with transaction info
  if (message->transaction_typename_size() == 0 && message->item_size() > 0) {
    message->add_transaction_typename(DefaultTransactionTypeName);

    for (auto& item : *message->mutable_item()) {
      item.clear_transaction_start_index();
      item.clear_transaction_typename_id();

      for (int i = 0; i < item.token_id_size(); ++i) {
        item.add_transaction_start_index(i);
        item.add_transaction_typename_id(0);
      }
      item.add_transaction_start_index(item.token_id_size());
    }
  }
}

template<>
inline void FixMessage(::artm::GetThetaMatrixArgs* message) {
  if (message->has_use_sparse_format()) {
    message->set_matrix_layout(MatrixLayout_Sparse);
  }
}

template<>
inline void FixMessage(::artm::GetTopicModelArgs* message) {
  if (message->has_use_sparse_format()) {
    message->set_matrix_layout(MatrixLayout_Sparse);
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
    for (int i = 0; i < size; ++i) {
      message->add_batch_weight(1.0f);
    }
  }

  for (int i = 0; i < message->batch_size(); ++i) {
    FixMessage(message->mutable_batch(i));
  }

  if (message->class_weight_size() == 0) {
    for (int i = 0; i < message->class_id_size(); ++i) {
      message->add_class_weight(1.0f);
    }
  }

  if (message->transaction_weight_size() == 0) {
    for (int i = 0; i < message->transaction_typename_size(); ++i) {
      message->add_transaction_weight(1.0f);
    }
  }
}

template<>
inline void FixMessage(::artm::TopTokensScoreConfig* message) {
  if (!message->has_class_id() || message->class_id().empty()) {
    message->set_class_id(DefaultClass);
  }
}

template<>
inline void FixMessage(::artm::MasterModelConfig* message) {
  if (message->class_weight_size() == 0) {
    for (int i = 0; i < message->class_id_size(); ++i) {
      message->add_class_weight(1.0f);
    }
  }

  if (message->transaction_weight_size() == 0) {
    for (int i = 0; i < message->transaction_typename_size(); ++i) {
      message->add_transaction_weight(1.0f);
    }
  }

  if (message->reuse_theta()) {
    message->set_cache_theta(true);
  }

  for (int i = 0; i < message->regularizer_config_size(); ++i) {
    FixMessage(message->mutable_regularizer_config(i));
  }

  for (int i = 0; i < message->score_config_size(); ++i) {
    ScoreConfig* score_config = message->mutable_score_config(i);
    FixMessage(score_config);

    if (!score_config->has_model_name()) {
      score_config->set_model_name(message->pwt_name());
    }
  }

  ScoreConfig* items_processed_score = message->add_score_config();
  items_processed_score->set_name("^^^ItemsProcessedScore^^^");
  items_processed_score->set_type(ScoreType_ItemsProcessed);
  items_processed_score->set_config(::artm::ItemsProcessedScore().SerializeAsString());

  if (message->topic_name_size() == 0) {
    message->set_ptd_name(std::string());
  }
}

template<>
inline void FixMessage(::artm::FitOfflineMasterModelArgs* message) {
  if (message->batch_weight_size() == 0) {
    for (int i = 0; i < message->batch_filename_size(); ++i) {
      message->add_batch_weight(1.0f);
    }
  }
}

template<>
inline void FixMessage(::artm::FitOnlineMasterModelArgs* message) {
  if (message->batch_weight_size() == 0) {
    for (int i = 0; i < message->batch_filename_size(); ++i) {
      message->add_batch_weight(1.0f);
    }
  }

  if (message->apply_weight_size() == 0) {
    for (int i = 0; i < message->decay_weight_size(); ++i) {
      message->add_apply_weight(1.0f - message->decay_weight(i));
    }
  }

  if (message->decay_weight_size() == 0) {
    for (int i = 0; i < message->apply_weight_size(); ++i) {
      message->add_decay_weight(1.0f - message->apply_weight(i));
    }
  }
}

template<>
inline void FixMessage(::artm::TransformMasterModelArgs* message) {
  for (int i = 0; i < message->batch_size(); ++i) {
    FixMessage(message->mutable_batch(i));
  }
}

template<>
inline void FixMessage(::artm::ImportBatchesArgs* message) {
  for (int i = 0; i < message->batch_size(); ++i) {
    FixMessage(message->mutable_batch(i));
  }
}

template<>
inline void FixMessage(::artm::ProcessBatchesResult* message) {
  for (int i = 0; i < message->score_data_size(); ++i) {
    FixMessage(message->mutable_score_data(i));
  }
}

template<>
inline void FixMessage(::artm::ScoreArray* message) {
  for (int i = 0; i < message->score_size(); ++i) {
    FixMessage(message->mutable_score(i));
  }
}

template<>
inline void FixMessage(::artm::MergeModelArgs* message) {
  if (message->source_weight().empty()) {
    for (int i = 0; i < message->nwt_source_name_size(); ++i) {
      message->add_source_weight(1.0f);
    }
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
  if (message.has_gamma()) {
    ss << "gamma:" << message.gamma() << ")";
  } else {
    ss << "gamma:None" << ")";
  }
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::InitializeModelArgs& message) {
  std::stringstream ss;
  ss << "InitializeModelArgs";
  ss << ": model_name=" << message.model_name();

  if (message.has_dictionary_name()) {
    ss << ", dictionary_name=" << message.dictionary_name();
  }
  ss << ", topic_name_size=" << message.topic_name_size();
  ss << ", seed=" << message.seed();
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::FilterDictionaryArgs& message) {
  std::stringstream ss;
  ss << "FilterDictionaryArgs";
  ss << ": dictionary_name=" << message.dictionary_name();

  if (message.has_class_id()) {
    ss << ", class_id=" << message.class_id();
  }
  if (message.has_min_df()) {
    ss << ", min_df=" << message.min_df();
  }
  if (message.has_max_df()) {
    ss << ", max_df=" << message.max_df();
  }
  if (message.has_min_tf()) {
    ss << ", min_tf=" << message.min_tf();
  }
  if (message.has_max_tf()) {
    ss << ", max_tf=" << message.max_tf();
  }

  if (message.has_min_df_rate()) {
    ss << ", min_df_rate=" << message.min_df_rate();
  }
  if (message.has_max_df_rate()) {
    ss << ", max_df_rate=" << message.max_df_rate();
  }

  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::GatherDictionaryArgs& message) {
  std::stringstream ss;
  ss << "GatherDictionaryArgs";
  ss << ": dictionary_target_name=" << message.dictionary_target_name();

  if (message.has_data_path()) {
    ss << ", data_path=" << message.data_path();
  }
  if (message.has_cooc_file_path()) {
    ss << ", cooc_file_path=" << message.cooc_file_path();
  }
  if (message.has_vocab_file_path()) {
    ss << ", vocab_file_path=" << message.vocab_file_path();
  }
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
  ss << ", num_document_passes=" << message.num_document_passes();
  for (int i = 0; i < message.regularizer_name_size(); ++i) {
    ss << ", regularizer=(name:" << message.regularizer_name(i) << ", tau:" << message.regularizer_tau(i) << ")";
  }
  ss << ", reuse_theta=" << (message.reuse_theta() ? "yes" : "no");
  ss << ", opt_for_avx=" << (message.opt_for_avx() ? "yes" : "no");
  ss << ", predict_class_id=" << (message.predict_class_id());
  for (int i = 0; i < message.transaction_typename_size(); ++i) {
    ss << ", transaction_typename=(" << message.transaction_typename(i)
       << ":" << message.transaction_weight(i) << ")";
  }
  ss << ", reset_nwt=" << (message.reset_nwt() ? "yes" : "no");
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
  for (int i = 0; i < message.nwt_source_name_size(); ++i) {
    ss << ", class=(" << message.nwt_source_name(i) << ":" << message.source_weight(i) << ")";
  }
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
  for (int i = 0; i < message.regularizer_settings_size(); ++i) {
    DescribeMessage(message.regularizer_settings(i));
  }
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::MasterModelConfig& message) {
  std::stringstream ss;
  ss << "MasterModelConfig";
  ss << ": topic_name_size=" << message.topic_name_size();
  ss << ", score_config_size=" << message.score_config_size();
  ss << ", num_processors=" << message.num_processors();
  ss << ", pwt_name=" << message.pwt_name();
  ss << ", nwt_name=" << message.nwt_name();
  ss << ", num_document_passes=" << message.num_document_passes();
  for (int i = 0; i < message.regularizer_config_size(); ++i) {
    ss << ", regularizer=("
       << message.regularizer_config(i).name() << ":"
       << message.regularizer_config(i).tau() << ")";
  }
  ss << ", reuse_theta=" << (message.reuse_theta() ? "yes" : "no");
  ss << ", cache_theta=" << (message.cache_theta() ? "yes" : "no");
  ss << ", opt_for_avx=" << (message.opt_for_avx() ? "yes" : "no");
  ss << ", disk_cache_path=" << message.disk_cache_path();
  for (int i = 0; i < message.transaction_typename_size(); ++i) {
    ss << ", transaction_type=(" << message.transaction_typename(i)
      << ":" << message.transaction_weight(i) << ")";
  }
  if (message.has_parent_master_model_id()) {
    ss << ", parent_master_model_id=" << message.parent_master_model_id();
    ss << ", parent_master_model_weight=" << message.parent_master_model_weight();
  }

  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::FitOfflineMasterModelArgs& message) {
  std::stringstream ss;
  ss << "FitOfflineMasterModelArgs";
  ss << ", batch_filename_size=" << message.batch_filename_size();
  ss << ", batch_weight_size=" << message.batch_weight_size();
  ss << ", num_collection_passes=" << message.num_collection_passes();
  ss << ", reset_nwt=" << (message.reset_nwt() ? "yes" : "no");
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
    if (i != 0) {
      ss << ", ";
    }
    ss << message.update_after(i) << ":";
    ss << message.apply_weight(i) << ":";
    ss << message.decay_weight(i);
  }
  ss << ")";
  ss << ", async=" << (message.async() ? "yes" : "no");
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::TransformMasterModelArgs& message) {
  std::stringstream ss;
  ss << "TransformMasterModelArgs";
  ss << ", batch_filename_size=" << message.batch_filename_size();
  ss << ", batch_size=" << message.batch_size();
  ss << ", theta_matrix_type=" << message.theta_matrix_type();
  ss << ", predict_class_id=" << (message.predict_class_id());
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::GetScoreValueArgs& message) {
  std::stringstream ss;
  ss << "GetScoreValueArgs";
  ss << ", score_name=" << message.score_name();
  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::ConfigureLoggingArgs& message) {
  std::stringstream ss;
  ss << "ConfigureLoggingArgs";
  ss << ", log_dir=" << (message.has_log_dir() ? message.log_dir() : "");

  ss << ", minloglevel=" << (message.has_minloglevel() ? to_string(message.minloglevel()) : "");
  ss << ", stderrthreshold=" << (message.has_stderrthreshold() ? to_string(message.stderrthreshold()) : "");

  ss << ", logtostderr=" << (message.has_logtostderr() ? (message.logtostderr() ? "yes" : "no") : "");
  ss << ", colorlogtostderr=" << (message.has_colorlogtostderr() ? (message.colorlogtostderr() ? "yes" : "no") : "");
  ss << ", alsologtostderr=" << (message.has_alsologtostderr() ? (message.alsologtostderr() ? "yes" : "no") : "");

  ss << ", logbufsecs=" << (message.has_logbufsecs() ? to_string(message.logbufsecs()) : "");
  ss << ", logbuflevel=" << (message.has_logbuflevel() ? to_string(message.logbuflevel()) : "");

  ss << ", max_log_size=" << (message.has_max_log_size() ? to_string(message.max_log_size()) : "");

  ss << ", stop_logging_if_full_disk=" <<
    (message.has_stop_logging_if_full_disk() ? (message.stop_logging_if_full_disk() ? "yes" : "no") : "");

  return ss.str();
}

template<>
inline std::string DescribeMessage(const ::artm::ItemsProcessedScore& message) {
  std::stringstream ss;
  ss << "ItemsProcessed";
  ss << ", num_items=" << message.value();
  ss << ", num_batches=" << message.num_batches();
  ss << ", token_weight=" << message.token_weight();
  ss << ", token_weight_in_effect=" << message.token_weight_in_effect();
  return ss.str();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Templates
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline bool ValidateMessage(const T& message, bool throw_error) {
  std::string ss = DescribeErrors(message);

  if (ss.empty()) {
    return true;
  }

  if (throw_error) {
    BOOST_THROW_EXCEPTION(InvalidOperation(ss));
  }

  LOG(WARNING) << ss;
  return false;
}

template <typename T>
inline bool FixAndValidateMessage(T* message, bool throw_error) {
  FixMessage(message);
  return ValidateMessage(*message, throw_error);
}

template<typename T>
inline void FixPackedMessage(std::string* message) {
  T config;
  if (config.ParseFromString(*message)) {
    FixMessage<T>(&config);
    config.SerializeToString(message);
  }
}

}  // namespace core
}  // namespace artm
