// Copyright 2017, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <cmath>
#include <algorithm>

#include "artm/core/dictionary.h"
#include "artm/core/exceptions.h"
#include "artm/core/phi_matrix_operations.h"
#include "artm/core/protobuf_helpers.h"
#include "artm/core/token.h"

#include "artm/score/topic_kernel.h"

namespace artm {
namespace score {

std::shared_ptr<Score> TopicKernel::CalculateScore(const artm::core::PhiMatrix& p_wt) {
  const int topic_size = p_wt.topic_size();
  const int token_size = p_wt.token_size();

  // parameters preparation
  std::shared_ptr<core::Dictionary> dictionary_ptr = nullptr;
  if (config_.has_cooccurrence_dictionary_name()) {
    dictionary_ptr = dictionary(config_.cooccurrence_dictionary_name());
  }
  bool count_coherence = dictionary_ptr != nullptr;

  const auto& topic_name = p_wt.topic_name();
  std::vector<bool> topics_to_score;
  if (config_.topic_name_size() == 0) {
    topics_to_score.assign(topic_size, true);
  } else {
    topics_to_score = core::is_member(topic_name, config_.topic_name());
  }

  ::artm::core::ClassId class_id = ::artm::core::DefaultClass;
  if (config_.has_class_id()) {
    class_id = config_.class_id();
  }

  float probability_mass_threshold = config_.probability_mass_threshold();
  if (probability_mass_threshold < 0 || probability_mass_threshold > 1) {
    BOOST_THROW_EXCEPTION(artm::core::ArgumentOutOfRangeException(
        "TopicKernelScoreConfig.probablility_mass_threshold",
        config_.probability_mass_threshold()));
  }

  // kernel scores calculation
  // the elements, that corresponds non-used topics, will have value (-1)
  TopicKernelScore* topic_kernel_score = new TopicKernelScore();
  std::shared_ptr<Score> retval(topic_kernel_score);
  auto kernel_size = topic_kernel_score->mutable_kernel_size();
  auto kernel_purity = topic_kernel_score->mutable_kernel_purity();
  auto kernel_contrast = topic_kernel_score->mutable_kernel_contrast();
  auto kernel_coherence = topic_kernel_score->mutable_coherence();
  float average_kernel_coherence = 0.0f;

  for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
    if (topics_to_score[topic_index]) {
      kernel_size->Add(0.0f);
      kernel_purity->Add(0.0f);
      kernel_contrast->Add(0.0f);
      kernel_coherence->Add(0.0f);

      topic_kernel_score->add_topic_name(topic_name.Get(topic_index));
    } else {
      kernel_size->Add(-1.0f);
      kernel_purity->Add(-1.0f);
      kernel_contrast->Add(-1.0f);
      kernel_coherence->Add(-1.0f);
    }
  }

  const auto& n_wt = GetPhiMatrix(instance_->config()->nwt_name());
  auto normalizers = artm::core::PhiMatrixOperations::FindNormalizers(*n_wt);

  auto norm_iter = normalizers.find(class_id);
  if (norm_iter == normalizers.end()) {
    BOOST_THROW_EXCEPTION(artm::core::InvalidOperation(
        "TopicKernelScoreConfig.class_id " + class_id +
        " does not exists in n_wt matrix"));
  }

  const auto& n_t = norm_iter->second;
  std::vector<std::vector<core::Token> > topic_kernel_tokens(
      topic_size, std::vector<core::Token>());

  for (int token_index = 0; token_index < token_size; ++token_index) {
    const auto& token = p_wt.token(token_index);
    if (token.class_id == class_id) {
      float p_w = 0.0;
      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
        if (topics_to_score[topic_index]) {
          p_w += p_wt.get(token_index, topic_index) * n_t[topic_index];
        }
      }

      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
        if (topics_to_score[topic_index]) {
          float value = p_wt.get(token_index, topic_index);
          float p_tw = (p_w > 0.0f) ? (value * n_t[topic_index] / p_w) : 0.0f;

          if (p_tw >= probability_mass_threshold) {
            artm::core::repeated_field_append(kernel_size, topic_index, 1.0f);
            artm::core::repeated_field_append(kernel_purity, topic_index, value);
            artm::core::repeated_field_append(kernel_contrast, topic_index, p_tw);
            topic_kernel_tokens[topic_index].push_back(p_wt.token(token_index));
          }
        }
      }
    }
  }

  // contrast = sum(p(t|w)) / kernel_size
  for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
    float value = 0.0f;
    if (kernel_size->Get(topic_index) > config_.eps()) {
      value = kernel_contrast->Get(topic_index) / kernel_size->Get(topic_index);
    }
    kernel_contrast->Set(topic_index, value);
  }

  if (count_coherence) {
    int denominator = 0;
    for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
      if (topics_to_score[topic_index]) {
        float value = dictionary_ptr->CountTopicCoherence(topic_kernel_tokens[topic_index]);
        artm::core::repeated_field_append(kernel_coherence, topic_index, value);
        average_kernel_coherence += value;
        ++denominator;
      }
    }
    average_kernel_coherence /= average_kernel_coherence > 0 ? denominator : 1;
  }

  float average_kernel_size = 0.0f;
  float average_kernel_purity = 0.0f;
  float average_kernel_contrast = 0.0f;
  float useful_topics_count = 0.0f;
  auto kernel_tokens = topic_kernel_score->mutable_kernel_tokens();

  for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
    float current_kernel_size = kernel_size->Get(topic_index);
    const bool useful_topic = (current_kernel_size != -1);
    if (useful_topic) {
      useful_topics_count += 1;
      average_kernel_size += current_kernel_size;
      average_kernel_purity += kernel_purity->Get(topic_index);
      average_kernel_contrast += kernel_contrast->Get(topic_index);

      StringArray* tokens = kernel_tokens->Add();
      for (unsigned token_id = 0; token_id < topic_kernel_tokens[topic_index].size(); ++token_id) {
        tokens->add_value(topic_kernel_tokens[topic_index][token_id].keyword);
      }
    }
  }
  average_kernel_size /= useful_topics_count;
  average_kernel_purity /= useful_topics_count;
  average_kernel_contrast /= useful_topics_count;

  topic_kernel_score->set_average_kernel_size(average_kernel_size);
  topic_kernel_score->set_average_kernel_purity(average_kernel_purity);
  topic_kernel_score->set_average_kernel_contrast(average_kernel_contrast);
  if (count_coherence) {
    topic_kernel_score->set_average_coherence(average_kernel_coherence);
  }

  return retval;
}

}  // namespace score
}  // namespace artm
