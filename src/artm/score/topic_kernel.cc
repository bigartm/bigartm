// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <cmath>
#include <algorithm>

#include "artm/core/exceptions.h"
#include "artm/core/protobuf_helpers.h"
#include "artm/core/topic_model.h"

#include "artm/score/coherence_plugin.h"
#include "artm/score/topic_kernel.h"

namespace artm {
namespace score {

std::shared_ptr<Score> TopicKernel::CalculateScore(const artm::core::TopicModel& topic_model) {
  int topics_count = topic_model.topic_size();
  int tokens_count = topic_model.token_size();

  // parameters preparation
  std::shared_ptr<core::Dictionary> dictionary_ptr = nullptr;
  if (config_.has_cooccurrence_dictionary_name())
    dictionary_ptr = dictionary(config_.cooccurrence_dictionary_name());
  bool count_coherence = dictionary_ptr != nullptr;

  auto topic_name = topic_model.topic_name();
  std::vector<bool> topics_to_score;
  if (config_.topic_name_size() > 0) {
    for (int i = 0; i < topics_count; ++i)
      topics_to_score.push_back(false);

    for (int topic_id = 0; topic_id < config_.topic_name_size(); ++topic_id) {
      for (int real_topic_id = 0; real_topic_id < topics_count; ++real_topic_id) {
        if (topic_name.Get(real_topic_id) == config_.topic_name(topic_id)) {
          topics_to_score[real_topic_id] = true;
          break;
        }
      }
    }
  } else {
    for (int i = 0; i < topics_count; ++i)
      topics_to_score.push_back(true);
  }

  ::artm::core::ClassId class_id = ::artm::core::DefaultClass;
  if (config_.has_class_id())
    class_id = config_.class_id();

  double probability_mass_threshold = config_.probability_mass_threshold();
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

  for (int topic_index = 0; topic_index < topics_count; ++topic_index) {
    if (topics_to_score[topic_index]) {
        kernel_size->add_value(0.0);
        kernel_purity->add_value(0.0);
        kernel_contrast->add_value(0.0);
        kernel_coherence->add_value(0.0);
    } else {
        kernel_size->add_value(-1);
        kernel_purity->add_value(-1);
        kernel_contrast->add_value(-1);
        kernel_coherence->add_value(-1);
    }
  }

  std::vector<std::vector<core::Token> > topic_kernel_tokens;
  for (int topic_index = 0; topic_index < topics_count; ++topic_index)
    topic_kernel_tokens.push_back(std::vector<core::Token>());

  for (int token_index = 0; token_index < tokens_count; token_index++) {
    if (topic_model.token(token_index).class_id == class_id) {
      ::artm::core::TopicWeightIterator topic_iter =
          topic_model.GetTopicWeightIterator(token_index);

      // calculate normalizer
      double normalizer = 0.0;
      while (topic_iter.NextTopic() < topics_count) {
        if (topics_to_score[topic_iter.TopicIndex()])
          normalizer += static_cast<double>(topic_iter.Weight());
      }
      topic_iter.Reset();
      while (topic_iter.NextTopic() < topics_count) {
        int topic_index = topic_iter.TopicIndex();
        if (topics_to_score[topic_index]) {
          double p_tw = (normalizer > 0.0) ? (topic_iter.Weight() / normalizer) : 0.0;

          if (p_tw >= probability_mass_threshold) {
            artm::core::repeated_field_append(kernel_size->mutable_value(), topic_index, 1.0);
            artm::core::repeated_field_append(kernel_purity->mutable_value(), topic_index,
                                              topic_iter.Weight());
            artm::core::repeated_field_append(kernel_contrast->mutable_value(), topic_index, p_tw);
            if (count_coherence)
              topic_kernel_tokens[topic_index].push_back(topic_model.token(token_index));
          }
        }
      }
    }
  }

  // contrast = sum(p(t|w)) / kernel_size
  for (int topic_index = 0; topic_index < topics_count; ++topic_index) {
    double value = 0;
    if (kernel_size->value(topic_index) > config_.eps()) {
      value = kernel_contrast->value(topic_index) / kernel_size->value(topic_index);
    }
    kernel_contrast->set_value(topic_index, value);
  }

  if (count_coherence) {
    int denominator = 0;
    for (int topic_index = 0; topic_index < topics_count; ++topic_index) {
      if (topics_to_score[topic_index]) {
        float value = CountTopicCoherence(dictionary_ptr, topic_kernel_tokens[topic_index]);
        artm::core::repeated_field_append(kernel_coherence->mutable_value(), topic_index, value);
        average_kernel_coherence += value;
        ++denominator;
      }
    }
    average_kernel_coherence /= average_kernel_coherence > 0 ? denominator : 1;
  }

  double average_kernel_size = 0.0;
  double average_kernel_purity = 0.0;
  double average_kernel_contrast = 0.0;
  double useful_topics_count = 0.0;
  for (int topic_index = 0; topic_index < topics_count; ++topic_index) {
    double current_kernel_size = kernel_size->value(topic_index);
    bool useful_topic = (current_kernel_size != -1);
    if (useful_topic) {
      useful_topics_count += 1;
      average_kernel_size += current_kernel_size;
      average_kernel_purity += kernel_purity->value(topic_index);
      average_kernel_contrast += kernel_contrast->value(topic_index);
    }
  }
  average_kernel_size /= useful_topics_count;
  average_kernel_purity /= useful_topics_count;
  average_kernel_contrast /= useful_topics_count;

  topic_kernel_score->set_average_kernel_size(average_kernel_size);
  topic_kernel_score->set_average_kernel_purity(average_kernel_purity);
  topic_kernel_score->set_average_kernel_contrast(average_kernel_contrast);
  topic_kernel_score->set_average_coherence(average_kernel_coherence);

  return retval;
}

}  // namespace score
}  // namespace artm
