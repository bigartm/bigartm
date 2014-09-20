// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <vector>
#include <string>

#include "artm/regularizer_sandbox/dirichlet_phi.h"
#include "artm/core/regularizable.h"

namespace artm {
namespace regularizer_sandbox {

bool DirichletPhi::RegularizePhi(::artm::core::Regularizable* topic_model, double tau) {
  bool has_dictionary = true;
  if (!config_.has_dictionary_name()) {
    has_dictionary = false;
  }

  auto dictionary_ptr = dictionary(config_.dictionary_name());
  if (has_dictionary && dictionary_ptr == nullptr) {
    has_dictionary = false;
  }

  if (!has_dictionary) {
    for (int topic_id = 0; topic_id < topic_model->topic_size(); ++topic_id) {
      for (int token_id = 0; token_id < topic_model->token_size(); ++token_id) {
        float value = static_cast<float>(tau * 1);
        topic_model->IncreaseRegularizerWeight(token_id, topic_id, value);
      }
    }
  } else {
    for (int topic_id = 0; topic_id < topic_model->topic_size(); ++topic_id) {
      for (int token_id = 0; token_id < topic_model->token_size(); ++token_id) {
        auto token = topic_model->token(token_id);

        float coef = 0;
        if (dictionary_ptr->find(token) != dictionary_ptr->end()) {
          coef = dictionary_ptr->find(token)->second.value();
        }

        float value = static_cast<float>(tau * coef);
        topic_model->IncreaseRegularizerWeight(token_id, topic_id, value);
      }
    }
  }
  return true;
}

bool DirichletPhi::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  DirichletPhiConfig regularizer_config;
  if (!regularizer_config.ParseFromArray(config_blob.c_str(), config_blob.length())) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse DirichletPhiConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  return true;
}

}  // namespace regularizer_sandbox
}  // namespace artm
