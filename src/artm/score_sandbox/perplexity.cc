// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/score_sandbox/perplexity.h"

#include <cmath>
#include <map>
#include <algorithm>

#include "artm/core/exceptions.h"
#include "artm/core/topic_model.h"

namespace artm {
namespace score_sandbox {

void Perplexity::AppendScore(
    const Item& item,
    const std::vector<artm::core::Token>& token_dict,
    const artm::core::TopicModel& topic_model,
    const artm::ModelConfig& model_config,
    const std::vector<float>& theta,
    Score* score) {
  int topics_count = topic_model.topic_size();

  // the following code counts sparsity of theta
  auto topic_name = topic_model.topic_name();
  std::vector<bool> topics_to_score;
  int topics_to_score_size = 0;

  if (config_.theta_sparsity_topic_name_size() > 0) {
    for (int i = 0; i < topics_count; ++i)
      topics_to_score.push_back(false);

    for (int topic_id = 0; topic_id < config_.theta_sparsity_topic_name_size(); ++topic_id) {
      for (int real_topic_id = 0; real_topic_id < topics_count; ++real_topic_id) {
        if (topic_name.Get(real_topic_id) == config_.theta_sparsity_topic_name(topic_id)) {
          topics_to_score[real_topic_id] = true;
          topics_to_score_size++;
          break;
        }
      }
    }
  } else {
    topics_to_score_size = topics_count;
    for (int i = 0; i < topics_count; ++i)
      topics_to_score.push_back(true);
  }

  int zero_topics_count = 0;
  for (int topic_index = 0; topic_index < topics_count; ++topic_index) {
    if ((fabs(theta[topic_index]) < config_.theta_sparsity_eps()) &&
        topics_to_score[topic_index]) {
      ++zero_topics_count;
    }
  }

  // the following code counts perplexity
  std::map<::artm::core::ClassId, float> class_weights;
  for (int i = 0; (i < model_config.class_id_size()) && (i < model_config.class_weight_size()); ++i)
    class_weights.insert(std::make_pair(model_config.class_id(i), model_config.class_weight(i)));
  bool use_class_id = !class_weights.empty();

  float n_d = 0;
  for (auto& field : item.field()) {
    for (int token_index = 0; token_index < field.token_count_size(); ++token_index) {
      float class_weight = 1.0f;
      if (use_class_id) {
        ::artm::core::ClassId class_id = token_dict[field.token_id(token_index)].class_id;
        auto iter = class_weights.find(class_id);
        if (iter == class_weights.end())
          continue;
        class_weight = iter->second;
      }

      n_d += class_weight * static_cast<float>(field.token_count(token_index));
    }
  }

  int zero_words = 0;
  double normalizer = 0;
  double raw = 0;

  bool has_dictionary = true;
  if (!config_.has_dictionary_name()) {
    has_dictionary = false;
  }

  auto dictionary_ptr = dictionary(config_.dictionary_name());
  if (has_dictionary && dictionary_ptr == nullptr) {
    has_dictionary = false;
  }

  bool use_document_unigram_model = true;
  if (config_.has_model_type()) {
    if (config_.model_type() == PerplexityScoreConfig_Type_UnigramCollectionModel) {
      if (has_dictionary) {
        use_document_unigram_model = false;
      } else {
        LOG(ERROR) << "Perplexity was configured to use UnigramCollectionModel with dictionary " <<
           config_.dictionary_name() << ". This dictionary can't be found.";
        return;
      }
    }
  }

  for (auto& field : item.field()) {
    for (int token_index = 0; token_index < field.token_count_size(); ++token_index) {
      double sum = 0.0;
      const artm::core::Token& token = token_dict[field.token_id(token_index)];

      float class_weight = 1.0f;
      if (use_class_id) {
        auto iter = class_weights.find(token.class_id);
        if (iter == class_weights.end())
          continue;
        class_weight = iter->second;
      }

      int token_count_int = field.token_count(token_index);
      if (token_count_int == 0) continue;
      double token_count = class_weight * static_cast<double>(token_count_int);

      if (topic_model.has_token(token)) {
        ::artm::core::TopicWeightIterator topic_iter = topic_model.GetTopicWeightIterator(token);
        while (topic_iter.NextNonZeroTopic() < topics_count) {
          sum += theta[topic_iter.TopicIndex()] * topic_iter.Weight();
        }
      }

      if (sum == 0.0) {
        if (use_document_unigram_model) {
          sum = token_count / n_d;
        } else {
          if (dictionary_ptr->find(token) != dictionary_ptr->end()) {
            float n_w = dictionary_ptr->find(token)->second.value();
            sum = n_w / dictionary_ptr->size();
          } else {
            LOG(INFO) << "No token " << token.keyword << " from class " << token.class_id <<
                "in dictionary, document unigram model will be used.";
            sum = token_count / n_d;
          }
        }
        zero_words++;
      }

      normalizer += token_count;
      raw        += token_count * log(sum);
    }
  }

  // prepare results
  PerplexityScore perplexity_score;
  perplexity_score.set_normalizer(normalizer);
  perplexity_score.set_raw(raw);
  perplexity_score.set_zero_words(zero_words);
  perplexity_score.set_theta_sparsity_zero_topics(zero_topics_count);
  perplexity_score.set_theta_sparsity_total_topics(topics_to_score_size);
  AppendScore(perplexity_score, score);
}

std::string Perplexity::stream_name() const {
  return config_.stream_name();
}

std::shared_ptr<Score> Perplexity::CreateScore() {
  return std::make_shared<PerplexityScore>();
}

void Perplexity::AppendScore(const Score& score, Score* target) {
  std::string error_message = "Unable downcast Score to PerplexityScore";
  const PerplexityScore* perplexity_score = dynamic_cast<const PerplexityScore*>(&score);
  if (perplexity_score == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(error_message));
  }

  PerplexityScore* perplexity_target = dynamic_cast<PerplexityScore*>(target);
  if (perplexity_target == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(error_message));
  }

  perplexity_target->set_normalizer(perplexity_target->normalizer() +
                                    perplexity_score->normalizer());
  perplexity_target->set_raw(perplexity_target->raw() +
                             perplexity_score->raw());
  perplexity_target->set_zero_words(perplexity_target->zero_words() +
                                    perplexity_score->zero_words());
  perplexity_target->set_value(exp(- perplexity_target->raw() / perplexity_target->normalizer()));
  perplexity_target->set_theta_sparsity_zero_topics(
      perplexity_target->theta_sparsity_zero_topics() +
      perplexity_score->theta_sparsity_zero_topics());
  perplexity_target->set_theta_sparsity_total_topics(
      perplexity_target->theta_sparsity_total_topics() +
      perplexity_score->theta_sparsity_total_topics());
  perplexity_target->set_theta_sparsity_value(
      static_cast<double>(perplexity_target->theta_sparsity_zero_topics()) /
      perplexity_target->theta_sparsity_total_topics());
}

}  // namespace score_sandbox
}  // namespace artm
