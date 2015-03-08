// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/score_sandbox/perplexity.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

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
  bool use_class_id = true;
  std::map<::artm::core::ClassId, float> class_weights;
  bool classes_from_config = false;
  for (int i = 0; i < config_.class_id_size(); ++i) {
    class_weights.insert(std::make_pair(config_.class_id(i), -1));
    classes_from_config = true;
  }

  int find_weights_count = 0;
  if (classes_from_config) {
    for (auto& class_weight : class_weights) {
      for (int i = 0; (i < model_config.class_id_size()) && (i < model_config.class_weight_size()); ++i) {
        if (model_config.class_id(i) == class_weight.first) {
          class_weight.second = model_config.class_weight(i);
          ++find_weights_count;
          break;
        }
      }
    }
    if (find_weights_count != class_weights.size()) {
      use_class_id = false;
      LOG(WARNING) << "Perplexity score: class_id provided through score config is unknown to model."
                   << " Default class with weight == 1 will be used for all tokens.";
    }
  } else {
    for (int i = 0; (i < model_config.class_id_size()) && (i < model_config.class_weight_size()); ++i)
      class_weights.insert(std::make_pair(model_config.class_id(i), model_config.class_weight(i)));
    use_class_id = !class_weights.empty();
    if (!use_class_id)
      LOG(WARNING) << "Perplexity score: no information about classes and their weights was found in model."
                   << " Default class with weight == 1 will be used for all tokens.";
  }

  const Field* field = nullptr;
  for (int field_index = 0; field_index < item.field_size(); field_index++) {
    if (item.field(field_index).name() == config_.field_name()) {
      field = &item.field(field_index);
    }
  }

  if (field == nullptr) {
    LOG(ERROR) << "Unable to find field " << config_.field_name() << " in item " << item.id();
    return;
  }

  std::vector<float> n_d;
  if (use_class_id) {
    n_d = std::vector<float>(class_weights.size(), 0.0f);
    for (int token_index = 0; token_index < field->token_count_size(); ++token_index) {
      int class_index = 0;
      for (auto& class_weight : class_weights) {
        if (class_weight.first == token_dict[field->token_id(token_index)].class_id) {
          n_d[class_index] += class_weight.second * static_cast<float>(field->token_count(token_index));
          break;
        }
        ++class_index;
      }
    }
  } else {
    n_d.push_back(0.0f);
    for (int token_index = 0; token_index < field->token_count_size(); ++token_index)
      n_d[0] += static_cast<float>(field->token_count(token_index));
  }

  std::vector<int> zero_words;
  std::vector<double> normalizer;
  std::vector<double> raw;
  if (use_class_id) {
    zero_words = std::vector<int>(class_weights.size(), 0);
    normalizer = std::vector<double>(class_weights.size(), 0.0);
    raw =        std::vector<double>(class_weights.size(), 0.0);
  } else {
    zero_words.push_back(0);
    normalizer.push_back(0.0);
    raw.push_back(0.0);
  }

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

  for (int token_index = 0; token_index < field->token_count_size(); ++token_index) {
    double sum = 0.0;
    const artm::core::Token& token = token_dict[field->token_id(token_index)];

    int token_count_int = field->token_count(token_index);
    if (token_count_int == 0) continue;
    double token_count = 0.0;

    int class_index = 0;
    if (use_class_id) {
      bool useless_token = true;
      for (auto& class_weight : class_weights) {
        if (class_weight.first == token.class_id) {
          token_count = class_weight.second * static_cast<double>(token_count_int);
          useless_token = false;
          break;
        }
        ++class_index;
      }
      if (useless_token) continue;
    } else {
      token_count = static_cast<double>(token_count_int);
    }

    if (topic_model.has_token(token)) {
      ::artm::core::TopicWeightIterator topic_iter = topic_model.GetTopicWeightIterator(token);
      while (topic_iter.NextNonZeroTopic() < topics_count) {
        sum += theta[topic_iter.TopicIndex()] * topic_iter.Weight();
      }
    }

    if (sum == 0.0) {
      if (use_document_unigram_model) {
        sum = token_count / n_d[class_index];
      } else {
        if (dictionary_ptr->find(token) != dictionary_ptr->end()) {
          float n_w = dictionary_ptr->find(token)->second.value();
          sum = n_w / dictionary_ptr->size();
        } else {
          LOG(INFO) << "No token " << token.keyword << " from class " << token.class_id <<
              "in dictionary, document unigram model will be used.";
          sum = token_count / n_d[class_index];
        }
      }
      zero_words[class_index]++;
    }

    normalizer[class_index] += token_count;
    raw[class_index]        += token_count * log(sum);
  }

  // prepare results
  PerplexityScore perplexity_score;
  if (use_class_id) {
    int class_index = 0;
    for (auto& class_weight : class_weights) {
      perplexity_score.add_normalizer(normalizer[class_index]);
      perplexity_score.add_raw(raw[class_index]);
      perplexity_score.add_zero_words(zero_words[class_index]);
      perplexity_score.add_class_id(class_weight.first);
      ++class_index;
    }
  } else {
      perplexity_score.add_normalizer(normalizer[0]);
      perplexity_score.add_raw(raw[0]);
      perplexity_score.add_zero_words(zero_words[0]);
      perplexity_score.add_class_id(artm::core::DefaultClass);
  }

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

  if (perplexity_target->class_id_size()) {
    // here we expects, that lists of class_ids in 'perplexity_score' and 'perplexity_target' are the same
    for (int class_index = 0; class_index < perplexity_score->class_id_size(); ++class_index) {
      perplexity_target->set_normalizer(class_index,
          perplexity_target->normalizer(class_index) + perplexity_score->normalizer(class_index));
      perplexity_target->set_raw(class_index,
          perplexity_target->raw(class_index) + perplexity_score->raw(class_index));
      perplexity_target->set_zero_words(class_index,
          perplexity_target->zero_words(class_index) + perplexity_score->zero_words(class_index));
      perplexity_target->set_value(class_index,
          exp(- perplexity_target->raw(class_index) / perplexity_target->normalizer(class_index)));
    }
  } else {
    // this case is for the first usage of 'perplexity_target'
    for (int class_index = 0; class_index < perplexity_score->class_id_size(); ++class_index) {
      perplexity_target->add_normalizer(perplexity_score->normalizer(class_index));
      perplexity_target->add_raw(perplexity_score->raw(class_index));
      perplexity_target->add_zero_words(perplexity_score->zero_words(class_index));
      perplexity_target->add_value(exp(- perplexity_target->raw(class_index) /
                                         perplexity_target->normalizer(class_index)));
      perplexity_target->add_class_id(perplexity_score->class_id(class_index));
    }
  }
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
