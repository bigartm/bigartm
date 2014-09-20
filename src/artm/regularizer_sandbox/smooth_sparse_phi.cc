// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <vector>
#include <string>

#include "artm/regularizer_sandbox/smooth_sparse_phi.h"
#include "artm/core/regularizable.h"

namespace artm {
namespace regularizer_sandbox {

bool SmoothSparsePhi::RegularizePhi(::artm::core::Regularizable* topic_model, double tau) {
  // read the parameters from config and control their correctness
  const int topic_size = topic_model->topic_size();
  int background_topics_count = 0;
  if (config_.has_background_topics_count()) {
    background_topics_count = config_.background_topics_count();
  }

  if (background_topics_count < 0 || background_topics_count > topic_size) {
    LOG(ERROR) << "Smooth/Sparse Phi: background_topics_count must be in [0, topics_size]";
    return false;
  }
  const int objective_topic_size = topic_size - background_topics_count;

  ::artm::FloatArray topics_coefficients;
  if (config_.has_topics_coefficients()) {
    topics_coefficients.CopyFrom(config_.topics_coefficients());
    if (topics_coefficients.value().size() != objective_topic_size) {
      LOG(ERROR) << "Smooth/Sparse Phi: len(topics_coefficients) must be equal" <<
        "to len(objective_topic_size)";
      return false;
    }
  } else {
    // make default values
    for (int topic_id = 0; topic_id < objective_topic_size; ++topic_id) {
      topics_coefficients.add_value(1);
    }
  }

  bool has_dictionary = true;
  if (!config_.has_dictionary_name()) {
    has_dictionary = false;
  }

  auto dictionary_ptr = dictionary(config_.dictionary_name());
  if (has_dictionary && dictionary_ptr == nullptr) {
    has_dictionary = false;
  }

  if (!has_dictionary) {
  // proceed the regularization
    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      if (topic_id < objective_topic_size) {
        for (int token_id = 0; token_id < topic_model->token_size(); ++token_id) {
          float value = static_cast<float>(tau * topics_coefficients.value().Get(topic_id) * (-1));
          topic_model->IncreaseRegularizerWeight(token_id, topic_id, value);
        }
      } else {
        for (int token_id = 0; token_id < topic_model->token_size(); ++token_id) {
          float value = static_cast<float>(tau * (+1));
          topic_model->IncreaseRegularizerWeight(token_id, topic_id, value);
        }
      }
    }
  } else {
    // proceed the regularization
    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      // objective topics
      if (topic_id < objective_topic_size) {
        for (int token_id = 0; token_id < topic_model->token_size(); ++token_id) {
          auto token = topic_model->token(token_id);

          float coef = 0;
          if (dictionary_ptr->find(token) != dictionary_ptr->end()) {
            coef = dictionary_ptr->find(token)->second.value();
          }

          float value = static_cast<float>(tau * topics_coefficients.value().Get(topic_id) * coef);
          topic_model->IncreaseRegularizerWeight(token_id, topic_id, value);
        }
      } else {  // background topics
        for (int token_id = 0; token_id < topic_model->token_size(); ++token_id) {
          auto token = topic_model->token(token_id);

          float coef = 0;
          if (dictionary_ptr->find(token) != dictionary_ptr->end()) {
            int index = topic_id - objective_topic_size;
            coef = dictionary_ptr->find(token)->second.values().value(index);
          }

          float value = static_cast<float>(tau * coef);
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
