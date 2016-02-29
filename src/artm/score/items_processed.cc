// Copyright 2014, Additive Regularization of Topic Models.

// Author: Marina Suvorova (m.dudarenko@gmail.com)

#include "artm/core/exceptions.h"

#include "artm/score/items_processed.h"

namespace artm {
namespace score {

void ItemsProcessed::AppendScore(
    const Item& item,
    const std::vector<artm::core::Token>& token_dict,
    const artm::core::PhiMatrix& p_wt,
    const artm::ProcessBatchesArgs& args,
    const std::vector<float>& theta,
    Score* score) {
  ItemsProcessedScore items_processed_score;
  items_processed_score.set_value(1);
  AppendScore(items_processed_score, score);
}

void ItemsProcessed::AppendScore(const Batch& batch, Score* score) {
  ItemsProcessedScore items_processed_score;
  items_processed_score.set_num_batches(1);
  AppendScore(items_processed_score, score);
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
  items_processed_target->set_num_batches(items_processed_target->num_batches() +
                                          items_processed_score->num_batches());
}

}  // namespace score
}  // namespace artm
