// Copyright 2014, Additive Regularization of Topic Models.

#include <algorithm>
#include <utility>

#include "artm/core/exceptions.h"
#include "artm/core/topic_model.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/score/coherency_plugin.h"
#include "artm/score/top_tokens.h"

namespace artm {
namespace score {

std::shared_ptr<Score> TopTokens::CalculateScore(const artm::core::TopicModel& topic_model) {
  int topics_size = topic_model.topic_size();
  int tokens_size = topic_model.token_size();

  std::shared_ptr<core::Dictionary> dictionary_ptr = nullptr;
  if (config_.has_cooccurrence_dictionary_name())
    dictionary_ptr = dictionary(config_.cooccurrence_dictionary_name());
  bool count_coherency = dictionary_ptr != nullptr;

  std::vector<int> topic_ids;
  google::protobuf::RepeatedPtrField<std::string> topic_name = topic_model.topic_name();
  if (config_.topic_name_size() == 0) {
    for (int i = 0; i < topics_size; ++i) {
      topic_ids.push_back(i);
    }
  } else {
    for (int i = 0; i < config_.topic_name_size(); ++i) {
      int index = ::artm::core::repeated_field_index_of(topic_name, config_.topic_name(i));
      if (index == -1) {
        BOOST_THROW_EXCEPTION(::artm::core::InvalidOperation(
          "Topic with name '" + config_.topic_name(i) + "' not found in the model"));
      }
      topic_ids.push_back(index);
    }
  }

  ::artm::core::ClassId class_id = ::artm::core::DefaultClass;
  if (config_.has_class_id())
    class_id = config_.class_id();

  std::vector<artm::core::Token> tokens;
  for (int token_index = 0; token_index < tokens_size; token_index++) {
    auto token = topic_model.token(token_index);
    if (token.class_id == class_id)
      tokens.push_back(token);
  }

  TopTokensScore* top_tokens_score = new TopTokensScore();
  std::shared_ptr<Score> retval(top_tokens_score);
  int num_entries = 0;

  float average_coherency = 0.0f;
  auto coherency = top_tokens_score->mutable_coherency();
  for (int i = 0; i < topic_ids.size(); ++i) {
    std::vector<std::pair<float, int>> p_wt;
    p_wt.reserve(tokens.size());

    for (int token_index = 0; token_index < tokens_size; token_index++) {
      auto token = topic_model.token(token_index);
      if (token.class_id != class_id)
        continue;

      ::artm::core::TopicWeightIterator topic_iter = topic_model.GetTopicWeightIterator(token);
      float weight = topic_iter[topic_ids[i]];

      p_wt.push_back(std::pair<float, int>(weight, p_wt.size()));
    }

    std::sort(p_wt.begin(), p_wt.end());

    int first_index = p_wt.size() - 1;
    int last_index = (p_wt.size() - config_.num_tokens());
    if (last_index < 0) last_index = 0;

    std::vector<core::Token> tokens_for_coherency;
    for (int token_index = first_index; token_index >= last_index; token_index--) {
      ::artm::core::Token token = tokens[p_wt[token_index].second];
      float weight = p_wt[token_index].first;
      top_tokens_score->add_token(token.keyword);
      top_tokens_score->add_weight(weight);
      top_tokens_score->add_topic_index(topic_ids[i]);
      top_tokens_score->add_topic_name(topic_name.Get(topic_ids[i]));
      num_entries++;

      if (count_coherency) tokens_for_coherency.push_back(token);
    }

    if (count_coherency) {
      float topic_coherency = CountTopicCoherency(dictionary_ptr, tokens_for_coherency);
      average_coherency += topic_coherency;
      coherency->add_value(topic_coherency);
    }
  }

  top_tokens_score->set_average_coherency(
    average_coherency > 0.0f ? average_coherency / topic_ids.size() : average_coherency);

  top_tokens_score->set_num_entries(num_entries);
  return retval;
}

}  // namespace score
}  // namespace artm
