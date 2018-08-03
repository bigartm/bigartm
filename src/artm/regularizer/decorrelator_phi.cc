// Copyright 2017, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <string>
#include <vector>
#include <utility>

#include "artm/core/protobuf_helpers.h"
#include "artm/core/phi_matrix.h"

#include "artm/regularizer/decorrelator_phi.h"

namespace artm {
namespace regularizer {

bool DecorrelatorPhi::RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                                    const ::artm::core::PhiMatrix& n_wt,
                                    ::artm::core::PhiMatrix* result) {
  // read the parameters from config and control their correctness
  const bool use_topic_pairs = (topic_pairs_.size() > 0);

  std::unordered_map<std::string, int> topics_to_regularize;
  if (!use_topic_pairs) {
    for (const auto& s : (config_.topic_name().size() ? config_.topic_name() : p_wt.topic_name())) {
      bool valid_topic = false;
      for (int i = 0; i < p_wt.topic_name().size(); ++i) {
        if (p_wt.topic_name(i) == s) {
          topics_to_regularize.insert(std::make_pair(s, i));
          valid_topic = true;
          break;
        }
      }
      if (!valid_topic) {
        LOG(WARNING) << "Topic name " << s << " is not presented into model and will be ignored";
      }
    }
  }

  std::unordered_map<std::string, int> all_topics;
  for (int i = 0; i < p_wt.topic_name().size(); ++i) {
    all_topics.insert(std::make_pair(p_wt.topic_name(i), i));
  }

  bool use_all_classes = false;
  if (config_.class_id_size() == 0) {
    use_all_classes = true;
  }

  // proceed the regularization
  for (int token_pwt_id = 0; token_pwt_id < p_wt.token_size(); ++token_pwt_id) {
    const auto& token = p_wt.token(token_pwt_id);
    if (!use_all_classes && !core::is_member(token.class_id, config_.class_id())) {
      continue;
    }

    int token_nwt_id = n_wt.token_index(token);
    if (token_nwt_id == -1) {
      continue;
    }

    // count sum of weights
    float weights_sum = 0.0f;

    // simple case (without topic_pairs)
    if (!use_topic_pairs) {
      // create general normalizer
      for (const auto& pair : topics_to_regularize) {
        weights_sum += p_wt.get(token_pwt_id, pair.second);
      }

      // process every topic from topic_names
      for (const auto& pair : topics_to_regularize) {
        float weight = p_wt.get(token_pwt_id, pair.second);
        float value = static_cast<float>(-weight * (weights_sum - weight));
        result->set(token_nwt_id, pair.second, value);
      }
    } else {  // complex case
      for (const auto& pair : topic_pairs_) {
        weights_sum = 0.0f;

        // check given topic exists in model
        auto first_iter = all_topics.find(pair.first);
        if (first_iter == all_topics.end()) {
          continue;
        }

        // create custom normilizer for this topic
        for (const auto& topic_and_value : pair.second) {
          auto second_iter = all_topics.find(topic_and_value.first);
          if (second_iter == all_topics.end()) {
            continue;
          }

          weights_sum += p_wt.get(token_pwt_id, second_iter->second) * topic_and_value.second;
        }

        // process this topic value
        float weight = p_wt.get(token_pwt_id, first_iter->second);
        float value = static_cast<float>(-weight * (weights_sum - weight));
        result->set(token_nwt_id, first_iter->second, value);
      }
    }
  }
  return true;
}

google::protobuf::RepeatedPtrField<std::string> DecorrelatorPhi::topics_to_regularize() {
  return config_.topic_name();
}

google::protobuf::RepeatedPtrField<std::string> DecorrelatorPhi::class_ids_to_regularize() {
  return config_.class_id();
}

void DecorrelatorPhi::UpdateTopicPairs(const DecorrelatorPhiConfig& config) {
  config_.clear_first_topic_name();
  config_.clear_second_topic_name();
  config_.clear_value();

  topic_pairs_.clear();
  int topics_len = config.first_topic_name_size();
  if (topics_len) {
    if (topics_len != config.second_topic_name_size() ||
      topics_len != config.value_size()) {
      BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
        "Both topic indices and value arrays should have the same length"));
    }

    for (int i = 0; i < topics_len; ++i) {
      std::string first_name = config.first_topic_name(i);
      auto iter = topic_pairs_.find(first_name);
      if (iter == topic_pairs_.end()) {
        topic_pairs_.insert(std::make_pair(first_name, std::unordered_map<std::string, float>()));
        iter = topic_pairs_.find(first_name);
      }
      iter->second.insert(std::make_pair(config.second_topic_name(i), config.value(i)));
    }
  }
}

bool DecorrelatorPhi::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  DecorrelatorPhiConfig regularizer_config;
  if (!regularizer_config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse DecorrelatorPhiConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  UpdateTopicPairs(regularizer_config);

  return true;
}

}  // namespace regularizer
}  // namespace artm
