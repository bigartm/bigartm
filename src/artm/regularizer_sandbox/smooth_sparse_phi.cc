// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <string>
#include <vector>

#include "artm/core/protobuf_helpers.h"
#include "artm/core/regularizable.h"
#include "artm/regularizer_sandbox/smooth_sparse_phi.h"

namespace artm {
namespace regularizer_sandbox {

bool SmoothSparsePhi::RegularizePhi(::artm::core::Regularizable* topic_model, double tau) {
  // read the parameters from config and control their correctness
  const int topic_size = topic_model->topic_size();
  const int token_size = topic_model->token_size();

  std::vector<bool> topics_to_regularize;
  if (config_.topic_name().size() == 0)
    topics_to_regularize.assign(topic_size, true);
  else
    topics_to_regularize = core::is_member(config_.topic_name(), topic_model->topic_name());

  bool use_all_classes = false;
  if (config_.class_id_size() == 0) {
    use_all_classes = true;
  }

  bool has_dictionary = true;
  if (!config_.has_dictionary_name()) {
    has_dictionary = false;
  }

  auto dictionary_ptr = dictionary(config_.dictionary_name());
  if (has_dictionary && dictionary_ptr == nullptr) {
    has_dictionary = false;
  }

  // fill the vector of coefficients with ones or with values from dictionary
  std::vector<float> coefficients;
  if (!has_dictionary) {
    for (int token_id = 0; token_id < token_size; ++token_id) {
      if (use_all_classes ||
          core::is_member(topic_model->token(token_id).class_id, config_.class_id()))
        coefficients.push_back(1);
    }
  } else {
    for (int token_id = 0; token_id < token_size; ++token_id) {
      auto token = topic_model->token(token_id);
      if (use_all_classes ||
          core::is_member(token.class_id, config_.class_id())) {
        auto entry_iter = dictionary_ptr->find(token);
        // don't process tokens without value in the dictionary
        if (entry_iter == dictionary_ptr->end())
          coefficients.push_back(0);
        else
          coefficients.push_back(entry_iter->second.value());
      }
    }
  }

  // proceed the regularization
  for (int token_id = 0; token_id < topic_model->token_size(); ++token_id) {
    bool use_this_token = core::is_member(topic_model->token(token_id).class_id, config_.class_id());
    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      if (topics_to_regularize[topic_id]) {
        if (use_all_classes || use_this_token) {
          float value = static_cast<float>(tau) * coefficients[token_id];
          topic_model->IncreaseRegularizerWeight(token_id, topic_id, value);
        }
      }
    }
  }

  return true;
}

bool SmoothSparsePhi::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  SmoothSparsePhiConfig regularizer_config;
  if (!regularizer_config.ParseFromArray(config_blob.c_str(), config_blob.length())) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse SmoothSparsePhiConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  return true;
}

}  // namespace regularizer_sandbox
}  // namespace artm
