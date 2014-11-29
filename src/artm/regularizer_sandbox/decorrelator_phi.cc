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
  auto class_name = topic_model->class_id();

  std::vector<bool> topics_to_regularize;
  std::vector<artm::core::ClassId> classes_to_regularize;

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

  if (config_.class_name_size() > 0) {
    for (auto& class_id : config_.class_name())
      classes_to_regularize.push_back(class_id);
  } else {
    classes_to_regularize = class_name;
  }

  // proceed the regularization
  for (int token_id = 0; token_id < topic_model->token_size(); ++token_id) {
    if (std::find(classes_to_regularize.begin(),
                  classes_to_regularize.end(),
                  topic_model->token(token_id).class_id)
        != classes_to_regularize.end()) {
      // count sum of weights
      auto topic_iterator = topic_model->GetTopicWeightIterator(token_id);
      float weights_sum = 0.0f;
      while (topic_iterator.NextTopic() < topic_size) {
        if (topics_to_regularize[topic_iterator.TopicIndex()])
          weights_sum += topic_iterator.Weight();
      }

      // form the value
      topic_iterator.Reset();
      while (topic_iterator.NextTopic() < topic_size) {
        int topic_id = topic_iterator.TopicIndex();
        if (topics_to_regularize[topic_id]) {
          float weight = topic_iterator.Weight();
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
