// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/score_sandbox/items_processed.h"

#include "artm/core/exceptions.h"
#include "artm/core/topic_model.h"

namespace artm {
namespace score_sandbox {

void ItemsProcessed::AppendScore(
    const Item& item,
    const std::vector<artm::core::Token>& token_dict,
    const artm::core::TopicModel& topic_model,
    const std::vector<float>& theta,
    Score* score) {
  ItemsProcessedScore items_processed_score;
  items_processed_score.set_value(items_processed_score.value() + 1);
  AppendScore(items_processed_score, score);
}

std::string ItemsProcessed::stream_name() const {
  return config_.stream_name();
}

std::shared_ptr<Score> ItemsProcessed::CreateScore() {
  return std::make_shared<ItemsProcessedScore>();
}

void ItemsProcessed::AppendScore(const Score& score, Score* target) {
  std::string error_message = "Unable downcast Score to SparsityThetaScore";
  const ItemsProcessedScore* items_processed_score = dynamic_cast<const ItemsProcessedScore*>(&score);
  if (items_processed_score == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(error_message));
  }

  ItemsProcessedScore* items_processed_target = dynamic_cast<ItemsProcessedScore*>(target);
  if (items_processed_target == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(error_message));
  }

  items_processed_target->set_value(items_processed_target->value() +
                                    items_processed_score->value());
}

}  // namespace score_sandbox
}  // namespace artm
