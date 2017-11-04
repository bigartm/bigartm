// Copyright 2017, Additive Regularization of Topic Models.

// Author: Marina Suvorova (m.dudarenko@gmail.com)

#include "artm/core/exceptions.h"

#include "artm/score/items_processed.h"
#include "artm/core/protobuf_helpers.h"

namespace artm {
namespace score {

void ItemsProcessed::AppendScore(const Batch& batch,
                                 const artm::core::PhiMatrix& p_wt,
                                 const artm::ProcessBatchesArgs& args,
                                 Score* score) {
  float token_weight = 0.0f;
  float token_weight_in_effect = 0.0f;

  for (const auto& item : batch.item()) {
    for (int token_index = 0; token_index < item.token_id_size(); token_index++) {
      int token_id = item.token_id(token_index);
      token_weight += item.token_weight(token_index);
      const std::string& token = batch.token(token_id);
      const std::string& class_id = batch.class_id(token_id);

      // Check whether token is in effect (e.g. present in the model, and belongs to relevant modality)
      if (!p_wt.has_token(::artm::core::Token(class_id, token))) {
        continue;
      }

      if (args.class_id_size() > 0 && !::artm::core::is_member(class_id, args.class_id())) {
        continue;
      }
      token_weight_in_effect += item.token_weight(token_index);
    }
  }

  ItemsProcessedScore items_processed_score;
  items_processed_score.set_num_batches(1);
  items_processed_score.set_value(batch.item_size());
  items_processed_score.set_token_weight(token_weight);
  items_processed_score.set_token_weight_in_effect(token_weight_in_effect);

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

  items_processed_target->set_value(
    items_processed_target->value() + items_processed_score->value());
  items_processed_target->set_num_batches(
    items_processed_target->num_batches() + items_processed_score->num_batches());
  items_processed_target->set_token_weight(
    items_processed_target->token_weight() + items_processed_score->token_weight());
  items_processed_target->set_token_weight_in_effect(
    items_processed_target->token_weight_in_effect() + items_processed_score->token_weight_in_effect());
}

}  // namespace score
}  // namespace artm
