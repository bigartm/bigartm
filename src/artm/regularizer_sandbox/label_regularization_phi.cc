// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include "artm/regularizer_sandbox/label_regularization_phi.h"

#include <map>
#include <string>
#include <vector>

#include "artm/core/protobuf_helpers.h"
#include "artm/core/regularizable.h"
#include "artm/core/topic_model.h"

namespace artm {
namespace regularizer_sandbox {

bool LabelRegularizationPhi::RegularizePhi(const ::artm::core::Regularizable& topic_model,
                                           ::artm::core::TokenCollectionWeights* result) {
  // read the parameters from config and control their correctness
  const int topic_size = topic_model.topic_size();
  const int token_size = topic_model.token_size();

  std::vector<bool> topics_to_regularize;
  if (config_.topic_name().size() == 0)
    topics_to_regularize.assign(topic_size, true);
  else
    topics_to_regularize = core::is_member(topic_model.topic_name(), config_.topic_name());

  bool use_all_classes = false;
  if (config_.class_id_size() == 0) {
    use_all_classes = true;
  }

  std::shared_ptr<core::Dictionary> dictionary_ptr = nullptr;
  if (config_.has_dictionary_name())
    dictionary_ptr = dictionary(config_.dictionary_name());
  bool has_dictionary = dictionary_ptr != nullptr;

  core::TokenCollectionWeights p_wt(topic_model.topic_size());
  topic_model.FindPwt(&p_wt);
  std::map<core::ClassId, std::vector<float> > n_t = topic_model.FindNormalizers();

  // proceed the regularization
  for (int token_id = 0; token_id < token_size; ++token_id) {
    auto token = topic_model.token(token_id);
    auto class_iter = n_t.find(token.class_id);
    assert(class_iter != n_t.end());

    float coefficient = 1.0f;
    if (has_dictionary) {
      if (use_all_classes ||
          core::is_member(token.class_id, config_.class_id())) {
        auto entry_ptr = dictionary_ptr->entry(token);
        // don't process tokens without value in the dictionary
        if (entry_ptr == nullptr) {
          coefficient = 0.0f;
        } else {
          if (entry_ptr->has_value())
            coefficient = entry_ptr->value();
        }
      }
    }

    if (!use_all_classes && !core::is_member(token.class_id, config_.class_id())) continue;

    // count sum of weights
    float weights_sum = 0.0f;
    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      if (topics_to_regularize[topic_id]) {
        // token_class_id is anyway presented in n_t
        weights_sum += p_wt[token_id][topic_id] * class_iter->second[topic_id];
      }
    }
    // form the value
    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      if (topics_to_regularize[topic_id]) {
        float weight = p_wt[token_id][topic_id] * class_iter->second[topic_id];
        float value = static_cast<float>(coefficient * weight / weights_sum);
        (*result)[token_id][topic_id] = value;
      }
    }
  }

  return true;
}

google::protobuf::RepeatedPtrField<std::string> LabelRegularizationPhi::topics_to_regularize() {
  return config_.topic_name();
}

google::protobuf::RepeatedPtrField<std::string> LabelRegularizationPhi::class_ids_to_regularize() {
  return config_.class_id();
}

bool LabelRegularizationPhi::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  LabelRegularizationPhiConfig regularizer_config;
  if (!regularizer_config.ParseFromArray(config_blob.c_str(), config_blob.length())) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse LabelRegularizationPhiConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  return true;
}

}  // namespace regularizer_sandbox
}  // namespace artm
