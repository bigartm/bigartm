// Copyright 2014, Additive Regularization of Topic Models.

#include <cmath>

#include "artm/core/exceptions.h"

#include "artm/score/sparsity_phi.h"

namespace artm {
namespace score {

std::shared_ptr<Score> SparsityPhi::CalculateScore(const artm::core::PhiMatrix& p_wt) {
  int topics_count = p_wt.topic_size();
  int tokens_count = p_wt.token_size();
  int zero_tokens_count = 0;
  int topics_to_score_size = 0;

  // parameters preparation
  auto topic_name = p_wt.topic_name();
  std::vector<bool> topics_to_score;
  if (config_.topic_name_size() > 0) {
    for (int i = 0; i < topics_count; ++i)
      topics_to_score.push_back(false);

    for (int topic_id = 0; topic_id < config_.topic_name_size(); ++topic_id) {
      for (int real_topic_id = 0; real_topic_id < topics_count; ++real_topic_id) {
        if (topic_name.Get(real_topic_id) == config_.topic_name(topic_id)) {
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

  ::artm::core::ClassId class_id = ::artm::core::DefaultClass;
  if (config_.has_class_id())
    class_id = config_.class_id();

  int class_tokens_count = 0;
  for (int token_index = 0; token_index < tokens_count; token_index++) {
    if (p_wt.token(token_index).class_id == class_id) {
      class_tokens_count++;
      for (int topic_index = 0; topic_index < topics_count; ++topic_index) {
        if ((fabs(p_wt.get(token_index, topic_index)) < config_.eps()) &&
            topics_to_score[topic_index]) {
          ++zero_tokens_count;
        }
      }
    }
  }

  SparsityPhiScore* sparsity_phi_score = new SparsityPhiScore();
  std::shared_ptr<Score> retval(sparsity_phi_score);

  sparsity_phi_score->set_zero_tokens(zero_tokens_count);
  sparsity_phi_score->set_total_tokens(class_tokens_count * topics_to_score_size);
  sparsity_phi_score->set_value(static_cast<double>(sparsity_phi_score->zero_tokens()) /
                                static_cast<double>(sparsity_phi_score->total_tokens()));

  return retval;
}

}  // namespace score
}  // namespace artm
