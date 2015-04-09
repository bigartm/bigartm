// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <map>
#include <string>
#include <vector>
#include <utility>

#include "artm/core/protobuf_helpers.h"
#include "artm/core/regularizable.h"
#include "artm/regularizer_sandbox/smooth_sparse_phi.h"

namespace artm {
namespace regularizer_sandbox {

bool SmoothSparsePhi::RegularizePhi(::artm::core::Regularizable* topic_model, double tau) {
  // read the parameters from config and control their correctness
  const int topic_size = topic_model->topic_size();
  const int token_size = topic_model->token_size();

  std::vector<bool> topics_to_regularize = core::is_member(config_.topic_name(),
                                                           topic_model->topic_name());

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

  std::map<core::Token, float> coeffs;
  if (!has_dictionary) {
    for (int token_id = 0; token_id < token_size; ++token_id) {
      auto token = topic_model->token(token_id);
      if (use_all_classes || core::is_member(token.class_id, config_.class_id()))
        coeffs.insert(std::pair<core::Token, float>(token, 1));
    }
  } else {
    for (auto entry_iter = dictionary_ptr->begin();
         entry_iter != dictionary_ptr->end();
         ++entry_iter) {
      auto token = entry_iter->first;
      if (use_all_classes || core::is_member(token.class_id, config_.class_id()))
        coeffs.insert(std::pair<core::Token, float>(token, entry_iter->second.value()));
    }
  }

  // proceed the regularization
  for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
    for (int token_id = 0; token_id < topic_model->token_size(); ++token_id) {
      if (topics_to_regularize[topic_id]) {
        auto token = topic_model->token(token_id);
        if (use_all_classes || core::is_member(token.class_id, config_.class_id())) {
          float value = static_cast<float>(tau) * coeffs.find(token)->second;
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
