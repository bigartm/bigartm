// Copyright 2017, Additive Regularization of Topic Models.

// Author: Marina Suvorova (m.dudarenko@gmail.com)

#include <cmath>

#include "artm/core/exceptions.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/score/sparsity_phi.h"

namespace artm {
namespace score {

std::shared_ptr<Score> SparsityPhi::CalculateScore(const artm::core::PhiMatrix& p_wt) {
  // parameters preparation
  const int topic_size = p_wt.topic_size();
  const int token_size = p_wt.token_size();
  ::google::protobuf::int64 zero_tokens_count = 0;

  std::vector<bool> topics_to_score;
  ::google::protobuf::int64 topics_to_score_size = topic_size;
  if (config_.topic_name_size() == 0) {
    topics_to_score.assign(topic_size, true);
  } else {
    topics_to_score = core::is_member(p_wt.topic_name(), config_.topic_name());
    topics_to_score_size = config_.topic_name_size();
  }

  ::artm::core::ClassId class_id = ::artm::core::DefaultClass;
  if (config_.has_class_id()) {
    class_id = config_.class_id();
  }

  ::google::protobuf::int64 class_tokens_count = 0;
  for (int token_index = 0; token_index < token_size; token_index++) {
    const auto& token = p_wt.token(token_index);
    if (token.class_id == class_id) {
      class_tokens_count++;
      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
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
  sparsity_phi_score->set_value(static_cast<float>(sparsity_phi_score->zero_tokens()) /
                                static_cast<float>(sparsity_phi_score->total_tokens()));

  return retval;
}

}  // namespace score
}  // namespace artm
