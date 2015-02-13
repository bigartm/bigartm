// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/score_sandbox/top_tokens.h"

#include <utility>
#include <algorithm>

#include "artm/core/exceptions.h"
#include "artm/core/topic_model.h"
#include "artm/core/protobuf_helpers.h"

namespace artm {
namespace score_sandbox {

std::shared_ptr<Score> TopTokens::CalculateScore(const artm::core::TopicModel& topic_model) {
  int topics_size = topic_model.topic_size();
  int tokens_size = topic_model.token_size();

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

  std::vector<std::vector<std::pair<float, artm::core::Token>>> p_wt;
  for (int topic_id : topic_ids) {
    p_wt.push_back(std::vector<std::pair<float, artm::core::Token>>());
  }

  ::artm::core::ClassId class_id = ::artm::core::DefaultClass;
  if (config_.has_class_id())
    class_id = config_.class_id();

  for (int token_index = 0; token_index < tokens_size; token_index++) {
    auto token = topic_model.token(token_index);
    if (token.class_id != class_id)
      continue;

    ::artm::core::TopicWeightIterator topic_iter = topic_model.GetTopicWeightIterator(token);
    for (int i = 0; i < topic_ids.size(); ++i) {
      float weight = topic_iter[topic_ids[i]];
      p_wt[i].push_back(std::pair<float, artm::core::Token>(weight, token));
    }
  }

  TopTokensScore* top_tokens_score = new TopTokensScore();
  std::shared_ptr<Score> retval(top_tokens_score);

  int num_entries = 0;
  for (int i = 0; i < topic_ids.size(); ++i) {
    std::sort(p_wt[i].begin(), p_wt[i].end());
    int first_index = p_wt[i].size() - 1;
    int last_index = (p_wt[i].size() - config_.num_tokens());
    if (last_index < 0) last_index = 0;
    for (int token_index = first_index; token_index >= last_index; token_index--) {
      ::artm::core::Token token = p_wt[i][token_index].second;
      float weight = p_wt[i][token_index].first;
      top_tokens_score->add_token(token.keyword);
      top_tokens_score->add_weight(weight);
      top_tokens_score->add_topic_index(topic_ids[i]);
      top_tokens_score->add_topic_name(topic_name.Get(topic_ids[i]));
      num_entries++;
    }
  }

  top_tokens_score->set_num_entries(num_entries);
  return retval;
}

}  // namespace score_sandbox
}  // namespace artm
