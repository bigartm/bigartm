// Copyright 2017, Additive Regularization of Topic Models.

// Author: Marina Suvorova (m.dudarenko@gmail.com)

#include "artm/core/exceptions.h"

#include "artm/score/items_processed.h"
#include "artm/core/protobuf_helpers.h"
#include "artm/core/common.h"

namespace artm {
namespace score {

void ItemsProcessed::AppendScore(const Batch& batch,
                                 const artm::core::PhiMatrix& p_wt,
                                 const artm::ProcessBatchesArgs& args,
                                 Score* score) {
  float token_weight = 0.0f;
  float token_weight_in_effect = 0.0f;

  for (const auto& item : batch.item()) {
    for (int token_index = 0; token_index < item.transaction_start_index_size(); ++token_index) {
      const int start_index = item.transaction_start_index(token_index);
      const int end_index = (token_index + 1) < item.transaction_start_index_size() ?
                            item.transaction_start_index(token_index + 1) :
                            item.transaction_token_id_size();
      std::string str;
      for (int token_id = start_index; token_id < end_index; ++token_id) {
        auto& tmp = batch.class_id(item.transaction_token_id(token_id));
        str += (token_id == start_index) ? tmp : artm::core::TransactionSeparator + tmp;
      }

      artm::core::TransactionType tt(str);
      if (args.transaction_type_size() > 0 && !tt.ContainsIn(args.transaction_type())) {
        continue;
      }

      bool all_transaction_tokens_exist = true;
      for (int idx = start_index; idx < end_index; ++idx) {
        const int token_id = item.transaction_token_id(idx);
        const std::string& token = batch.token(token_id);
        const std::string& class_id = batch.class_id(token_id);

        // Check whether token is in effect,
        // e.g. present in the model, and belongs to relevant modality and tt)
        if (!p_wt.has_token(::artm::core::Token(class_id, token, tt))) {
          all_transaction_tokens_exist = false;
          break;
        }
      }

      if (all_transaction_tokens_exist) {
        token_weight += item.token_weight(token_index);
        token_weight_in_effect += item.token_weight(token_index);
      }
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
