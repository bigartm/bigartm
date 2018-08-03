// Copyright 2017, Additive Regularization of Topic Models.

// Author: Alexander Frey (sashafrey@gmail.com)

#include <cmath>

#include "artm/core/common.h"
#include "artm/core/exceptions.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/score/class_precision.h"

namespace artm {
namespace score {

void ClassPrecision::AppendScore(
    const Item& item,
    const std::vector<artm::core::Token>& token_dict,
    const artm::core::PhiMatrix& p_wt,
    const artm::ProcessBatchesArgs& args,
    const std::vector<float>& theta,
    Score* score) {
  if (!args.has_predict_class_id()) {
    return;
  }

  const int topic_size = p_wt.topic_size();

  float max_token_weight = 0.0f;
  std::string keyword;
  for (int token_index = 0; token_index < p_wt.token_size(); token_index++) {
    const auto& token = p_wt.token(token_index);
    if (token.class_id != args.predict_class_id()) {
      continue;
    }

    float weight = 0.0f;
    for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
      weight += theta[topic_index] * p_wt.get(token_index, topic_index);
    }

    if (weight >= max_token_weight) {
      keyword = token.keyword;
      max_token_weight = weight;
    }
  }

  bool error = true;
  for (int t_index = 0; t_index < item.transaction_start_index_size() - 1; ++t_index) {
    const int start_index = item.transaction_start_index(t_index);
    const int end_index = item.transaction_start_index(t_index + 1);

    for (int token_id = start_index; token_id < end_index; ++token_id) {
      const auto& token = token_dict[item.token_id(token_id)];
      if (token.class_id == args.predict_class_id() && token.keyword == keyword) {
        error = false;
        break;
      }
    }
  }

  ClassPrecisionScore class_prediction_score;
  class_prediction_score.set_error(error ? 1.0f : 0.0f);
  class_prediction_score.set_total(1.0f);
  AppendScore(class_prediction_score, score);
}

std::shared_ptr<Score> ClassPrecision::CreateScore() {
  return std::make_shared<ClassPrecisionScore>();
}

void ClassPrecision::AppendScore(const Score& score, Score* target) {
  std::string error_message = "Unable downcast Score to ClassPrecisionScore";
  const ClassPrecisionScore* class_precision_score = dynamic_cast<const ClassPrecisionScore*>(&score);
  if (class_precision_score == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(error_message));
  }

  ClassPrecisionScore* class_precision_target = dynamic_cast<ClassPrecisionScore*>(target);
  if (class_precision_target == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(error_message));
  }

  class_precision_target->set_error(class_precision_target->error() +
                                    class_precision_score->error());
  class_precision_target->set_total(class_precision_target->total() +
                                    class_precision_score->total());
  class_precision_target->set_value(1.0f - class_precision_target->error() /
                                           class_precision_target->total());
}

}  // namespace score
}  // namespace artm
