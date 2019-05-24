// Copyright 2017, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <string>
#include <vector>

#include "artm/core/protobuf_helpers.h"
#include "artm/core/phi_matrix.h"
#include "artm/core/phi_matrix_operations.h"
#include "artm/regularizer/smooth_time_in_topics_phi.h"

namespace artm {
namespace regularizer {

bool SmoothTimeInTopicsPhi::RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                                          const ::artm::core::PhiMatrix& n_wt,
                                          ::artm::core::PhiMatrix* result) {
  if (!::artm::core::PhiMatrixOperations::HasEqualShape(p_wt, n_wt)) {
    LOG(ERROR) << "SmoothTimeInTopicsPhi does not support changes in p_wt and n_wt matrix. Cancel it's launch.";
    return false;
  }
  // read the parameters from config and control their correctness
  const int topic_size = p_wt.topic_size();
  const int token_size = p_wt.token_size();

  std::vector<bool> topics_to_regularize;
  if (config_.topic_name().size() == 0) {
    topics_to_regularize.assign(topic_size, true);
  } else {
    topics_to_regularize = core::is_member(p_wt.topic_name(), config_.topic_name());
  }

  const auto& class_id = config_.class_id();

  // proceed the regularization
  // will update only tokens of given modality, that have prev and post tokens of this modality
  int index_prev_prev = -1;
  int index_prev = -1;
  for (int token_id = 0; token_id < token_size; ++token_id) {
    const auto& token = p_wt.token(token_id);

    if (token.class_id != class_id) {
      continue;
    }

    if (index_prev_prev < 0) {
      index_prev_prev = token_id;
      continue;
    }

    if (index_prev < 0) {
      index_prev = token_id;
      continue;
    }

    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      if (topics_to_regularize[topic_id]) {
        float value = p_wt.get(index_prev, topic_id);

        value *= ((p_wt.get(index_prev_prev, topic_id) - value) > 0.0 ? 1.0 : -1.0) +
                 ((p_wt.get(token_id, topic_id) - value) > 0.0 ? 1.0 : -1.0);

        result->set(index_prev, topic_id, value);
      }
    }
    index_prev_prev = index_prev;
    index_prev = token_id;
  }

  return true;
}

google::protobuf::RepeatedPtrField<std::string> SmoothTimeInTopicsPhi::topics_to_regularize() {
  return config_.topic_name();
}

google::protobuf::RepeatedPtrField<std::string> SmoothTimeInTopicsPhi::class_ids_to_regularize() {
  google::protobuf::RepeatedPtrField<std::string> retval;
  std::string* ptr = retval.Add();
  *ptr = config_.class_id();
  return retval;
}

bool SmoothTimeInTopicsPhi::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  SmoothTimeInTopicsPhiConfig regularizer_config;
  if (!regularizer_config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse SmoothSparsePhiConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  return true;
}

}  // namespace regularizer
}  // namespace artm
