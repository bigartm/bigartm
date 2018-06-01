// Copyright 2017, Additive Regularization of Topic Models.

// Author: Marina Suvorova (m.dudarenko@gmail.com)

#include "artm/core/exceptions.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/score/theta_snippet.h"

namespace artm {
namespace score {

void ThetaSnippet::AppendScore(
    const Item& item,
    const Batch& batch,
    const std::vector<artm::core::Token>& token_dict,
    const artm::core::PhiMatrix& p_wt,
    const artm::ProcessBatchesArgs& args,
    const std::vector<float>& theta,
    Score* score) {
  const int topic_size = p_wt.topic_size();

  ThetaSnippetScore theta_snippet_score;
  theta_snippet_score.add_item_id(item.id());
  auto theta_snippet_item = theta_snippet_score.add_values();
  for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
    theta_snippet_item->add_value(theta[topic_index]);
  }

  AppendScore(theta_snippet_score, score);
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

  if (config_.num_items() <= 0 || theta_snippet_score->values_size() == 0) {
    return;
  }

  while (theta_snippet_target->values_size() < config_.num_items()) {
    theta_snippet_target->add_item_id(-1);
    artm::FloatArray* values_target = theta_snippet_target->add_values();
    for (int i = 0; i < theta_snippet_score->values(0).value_size(); ++i) {
      values_target->add_value(0.0f);
    }
  }

  for (int item_index = 0; item_index < theta_snippet_score->item_id_size(); item_index++) {
    int item_id = theta_snippet_score->item_id(item_index);
    if (item_id < 0) {
      continue;
    }

    theta_snippet_target->set_item_id(item_id % config_.num_items(), item_id);
    artm::FloatArray* values_target = theta_snippet_target->mutable_values(item_id % config_.num_items());
    values_target->Clear();

    auto theta_snippet_item_score = theta_snippet_score->values(item_index);
    for (int topic_index = 0; topic_index < theta_snippet_item_score.value_size(); topic_index++) {
      values_target->add_value(theta_snippet_item_score.value(topic_index));
    }
  }
}

}  // namespace score
}  // namespace artm
