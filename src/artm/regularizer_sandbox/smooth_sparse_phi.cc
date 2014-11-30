// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <string>
#include <vector>

#include "artm/core/regularizable.h"
#include "artm/regularizer_sandbox/smooth_sparse_phi.h"

namespace artm {
namespace regularizer_sandbox {

bool SmoothSparsePhi::RegularizePhi(::artm::core::Regularizable* topic_model, double tau) {
  // read the parameters from config and control their correctness
  const int topic_size = topic_model->topic_size();

  auto topic_name = topic_model->topic_name();
  std::vector<bool> topics_to_regularize;
  if (config_.topic_name_size() > 0) {
    for (int i = 0; i < topic_size; ++i)
      topics_to_regularize.push_back(false);

    for (int topic_id = 0; topic_id < config_.topic_name_size(); ++topic_id) {
      for (int real_topic_id = 0; real_topic_id < topic_size; ++real_topic_id) {
        if (topic_name.Get(real_topic_id) == config_.topic_name(topic_id)) {
          topics_to_regularize[real_topic_id] = true;
          break;
        }
      }
    }
  } else {
    for (int i = 0; i < topic_size; ++i)
      topics_to_regularize.push_back(true);
  }

  bool use_all_classes = false;
  std::vector<artm::core::ClassId> classes_to_regularize;
  if (config_.class_name_size() > 0) {
    for (auto& class_id : config_.class_name())
      classes_to_regularize.push_back(class_id);
  } else {
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

  if (!has_dictionary) {
  // proceed the regularization
    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      for (int token_id = 0; token_id < topic_model->token_size(); ++token_id) {
        if (topics_to_regularize[topic_id]) {
          bool regularize_this_token = false;
          if (!use_all_classes) {
            if (std::find(classes_to_regularize.begin(),
                          classes_to_regularize.end(),
                          topic_model->token(token_id).class_id)
                != classes_to_regularize.end()) {
              regularize_this_token = true;
            }
          } else {
            regularize_this_token = true;
          }
          if (regularize_this_token) {
            topic_model->IncreaseRegularizerWeight(token_id, topic_id, static_cast<float>(tau));
          }
        }
      }
    }
  } else {
    // proceed the regularization
    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      for (int token_id = 0; token_id < topic_model->token_size(); ++token_id) {
        auto token = topic_model->token(token_id);
        if (topics_to_regularize[topic_id]) {
          bool regularize_this_token = false;
          if (!use_all_classes) {
            if (std::find(classes_to_regularize.begin(),
                          classes_to_regularize.end(),
                          topic_model->token(token_id).class_id)
                != classes_to_regularize.end()) {
              regularize_this_token = true;
            }
          } else {
            regularize_this_token = true;
          }
          if (regularize_this_token) {
            if (dictionary_ptr->find(token) != dictionary_ptr->end()) {
              float value = dictionary_ptr->find(token)->second.value() * static_cast<float>(tau);
              topic_model->IncreaseRegularizerWeight(token_id, topic_id, value);
            }
          }
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
