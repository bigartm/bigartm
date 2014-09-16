// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/score_sandbox/theta_snippet.h"

#include "artm/core/exceptions.h"
#include "artm/core/topic_model.h"

namespace artm {
namespace score_sandbox {

void ThetaSnippet::AppendScore(
    const Item& item,
    const std::vector<artm::core::Token>& token_dict,
    const artm::core::TopicModel& topic_model,
    const std::vector<float>& theta,
    Score* score) {
  int topics_size = topic_model.topic_size();

  ThetaSnippetScore theta_snippet_score;

  for (int item_index = 0; item_index < config_.item_id_size(); item_index++) {
    if (item_index == item.id()) {
      theta_snippet_score.add_item_id(item_index);
      auto theta_snippet_item = theta_snippet_score.add_values();
      for (int topic_index = 0; topic_index < topics_size; ++topic_index) {
        theta_snippet_item->add_value(theta[topic_index]);
      }
      break;
    }
  }

  AppendScore(theta_snippet_score, score);
}

std::string ThetaSnippet::stream_name() const {
  return config_.stream_name();
}

std::shared_ptr<Score> ThetaSnippet::CreateScore() {
  return std::make_shared<ThetaSnippetScore>();
}

void ThetaSnippet::AppendScore(const Score& score, Score* target) {
  std::string error_message = "Unable downcast Score to SparsityThetaScore";
  const ThetaSnippetScore* theta_snippet_score = dynamic_cast<const ThetaSnippetScore*>(&score);
  if (theta_snippet_score == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(error_message));
  }

  ThetaSnippetScore* theta_snippet_target = dynamic_cast<ThetaSnippetScore*>(target);
  if (theta_snippet_target == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(error_message));
  }

  for (int item_index = 0; item_index < theta_snippet_score->item_id_size(); item_index++) {
    theta_snippet_target->add_item_id(theta_snippet_score->item_id().Get(item_index));
    auto theta_snippet_item_target = theta_snippet_target->add_values();
    auto theta_snippet_item_score = theta_snippet_score->values(item_index);
    for (int topic_index = 0; topic_index < theta_snippet_item_score.value_size(); topic_index++) {
      theta_snippet_item_target->add_value(theta_snippet_item_score.value(topic_index));
    }
  }
}

}  // namespace score_sandbox
}  // namespace artm
