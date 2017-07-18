// Copyright 2017, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <string>
#include <vector>

#include "artm/core/protobuf_helpers.h"
#include "artm/core/phi_matrix.h"
#include "artm/regularizer/smooth_time_in_topics_phi.h"

namespace artm {
namespace regularizer {

bool SmoothTimeInTopicsPhi::RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                                          const ::artm::core::PhiMatrix& n_wt,
                                          ::artm::core::PhiMatrix* result) {
  // read the parameters from config and control their correctness
  const int topic_size = p_wt.topic_size();
  const int token_size = p_wt.token_size();

  std::vector<bool> topics_to_regularize;
  if (config_.topic_name().size() == 0)
    topics_to_regularize.assign(topic_size, true);
  else
    topics_to_regularize = core::is_member(p_wt.topic_name(), config_.topic_name());

  bool use_all_classes = false;
  if (config_.class_id_size() == 0) {
    use_all_classes = true;
  }

  // proceed the regularization
  for (int token_id = 1; token_id < token_size - 1; ++token_id) {
    const ::artm::core::Token& token = p_wt.token(token_id);

    if (!use_all_classes && !core::is_member(token.class_id, config_.class_id())) continue;

    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      if (topics_to_regularize[topic_id]) {
        double value = p_wt.get(token_id, topic_id);

        value *= ((p_wt.get(token_id - 1, topic_id) - value) > 0.0 ? 1.0 : -1.0) +
                 ((p_wt.get(token_id + 1, topic_id) - value) > 0.0 ? 1.0 : -1.0);

        result->set(token_id, topic_id, value);
      }
    }
  }

  return true;
}

google::protobuf::RepeatedPtrField<std::string> SmoothTimeInTopicsPhi::topics_to_regularize() {
  return config_.topic_name();
}

google::protobuf::RepeatedPtrField<std::string> SmoothTimeInTopicsPhi::class_ids_to_regularize() {
  return config_.class_id();
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
