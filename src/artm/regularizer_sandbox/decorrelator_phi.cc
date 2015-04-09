// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include "artm/regularizer_sandbox/decorrelator_phi.h"

#include <string>
#include <vector>

#include "artm/core/regularizable.h"
#include "artm/core/topic_model.h"

namespace artm {
namespace regularizer_sandbox {

bool DecorrelatorPhi::RegularizePhi(::artm::core::Regularizable* topic_model, double tau) {
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
  if (config_.class_id_size() > 0) {
    for (auto& class_id : config_.class_id())
      classes_to_regularize.push_back(class_id);
  } else {
    use_all_classes = true;
  }

  ::artm::core::TokenCollectionWeights p_wt(topic_model->topic_size());
  topic_model->FindPwt(&p_wt);

  // proceed the regularization
  for (int token_id = 0; token_id < topic_model->token_size(); ++token_id) {
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
      // count sum of weights
      float weights_sum = 0.0f;
      for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
        if (topics_to_regularize[topic_id])
          weights_sum += p_wt[token_id][topic_id];
      }

      // form the value
      for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
        if (topics_to_regularize[topic_id]) {
          float weight = p_wt[token_id][topic_id];
          float value = static_cast<float>(- tau * weight * (weights_sum - weight));
          topic_model->IncreaseRegularizerWeight(token_id, topic_id, value);
        }
      }
    }
  }
  return true;
}

bool DecorrelatorPhi::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  DecorrelatorPhiConfig regularizer_config;
  if (!regularizer_config.ParseFromArray(config_blob.c_str(), config_blob.length())) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse DecorrelatorPhiConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  return true;
}

}  // namespace regularizer_sandbox
}  // namespace artm
