// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include "artm/regularizer_sandbox/decorrelator_phi.h"

#include <string>

#include "artm/core/regularizable.h"
#include "artm/core/topic_model.h"

namespace artm {
namespace regularizer_sandbox {

bool DecorrelatorPhi::RegularizePhi(::artm::core::Regularizable* topic_model, double tau) {
  // read the parameters from config and control their correctness
  const int topic_size = topic_model->topic_size();

  ::artm::BoolArray topics_to_regularize;
  if (config_.has_topics_to_regularize()) {
    topics_to_regularize.CopyFrom(config_.topics_to_regularize());
    if (topics_to_regularize.value().size() != topic_size) {
      LOG(ERROR) << "Decorrelator Phi: len(topics_to_regularize) must be equal to len(topic_size)";
      return false;
    }
  } else {
    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      topics_to_regularize.add_value(true);
    }
  }
  // proceed the regularization
  for (int token_id = 0; token_id < topic_model->token_size(); ++token_id) {
    // count sum of weights
    auto topic_iterator = topic_model->GetTopicWeightIterator(token_id);
    float weights_sum = 0;
    while (topic_iterator.NextTopic() < topic_size) {
      if (topics_to_regularize.value(topic_iterator.TopicIndex())) {
        weights_sum += topic_iterator.Weight();
      }
    }
    // form the value
    topic_iterator.Reset();
    while (topic_iterator.NextTopic() < topic_size) {
      int topic_id = topic_iterator.TopicIndex();
      if (topics_to_regularize.value(topic_id)) {
        float weight = topic_iterator.Weight();
        float value = static_cast<float>(- tau * weight * (weights_sum - weight));
        topic_model->IncreaseRegularizerWeight(token_id, topic_id, value);
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
