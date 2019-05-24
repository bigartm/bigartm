// Copyright 2017, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <string>
#include <vector>

#include "artm/core/protobuf_helpers.h"
#include "artm/core/phi_matrix.h"
#include "artm/regularizer/smooth_sparse_phi.h"

namespace artm {
namespace regularizer {

SmoothSparsePhi::SmoothSparsePhi(const SmoothSparsePhiConfig& config)
    : config_(config)
    , transform_function_(nullptr) {
  if (config.has_transform_config()) {
    transform_function_ = artm::core::TransformFunction::create(config.transform_config());
  } else {
    transform_function_ = artm::core::TransformFunction::create();
  }
}

bool SmoothSparsePhi::RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                                    const ::artm::core::PhiMatrix& n_wt,
                                    ::artm::core::PhiMatrix* result) {
  // read the parameters from config and control their correctness
  const int topic_size = p_wt.topic_size();

  std::vector<bool> topics_to_regularize;
  if (config_.topic_name().size() == 0) {
    topics_to_regularize.assign(topic_size, true);
  } else {
    topics_to_regularize = core::is_member(p_wt.topic_name(), config_.topic_name());
  }

  bool use_all_classes = false;
  if (config_.class_id_size() == 0) {
    use_all_classes = true;
  }

  std::shared_ptr<core::Dictionary> dictionary_ptr = nullptr;
  if (config_.has_dictionary_name()) {
    dictionary_ptr = dictionary(config_.dictionary_name());
  }

  // proceed the regularization
  for (int token_pwt_id = 0; token_pwt_id < p_wt.token_size(); ++token_pwt_id) {
    float coefficient = 1.0f;
    const auto& token = p_wt.token(token_pwt_id);

    if (!use_all_classes && !core::is_member(token.class_id, config_.class_id())) {
      continue;
    }

    if (dictionary_ptr != nullptr) {
      auto entry_ptr = dictionary_ptr->entry(token);

      // don't process tokens without value in the dictionary
      if (entry_ptr == nullptr) {
        continue;
      }

      coefficient = entry_ptr->token_value();
    }

    int token_nwt_id = n_wt.token_index(token);
    if (token_nwt_id == -1) {
      continue;
    }

    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      if (topics_to_regularize[topic_id]) {
        float value = transform_function_->apply(p_wt.get(token_pwt_id, topic_id));
        result->set(token_nwt_id, topic_id, coefficient * value);
      }
    }
  }

  return true;
}

google::protobuf::RepeatedPtrField<std::string> SmoothSparsePhi::topics_to_regularize() {
  return config_.topic_name();
}

google::protobuf::RepeatedPtrField<std::string> SmoothSparsePhi::class_ids_to_regularize() {
  return config_.class_id();
}

bool SmoothSparsePhi::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  SmoothSparsePhiConfig regularizer_config;
  if (!regularizer_config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse SmoothSparsePhiConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);

  if (config_.has_transform_config()) {
    transform_function_ = artm::core::TransformFunction::create(config_.transform_config());
  } else {
    transform_function_ = artm::core::TransformFunction::create();
  }

  return true;
}

}  // namespace regularizer
}  // namespace artm
