// Copyright 2017, Additive Regularization of Topic Models.

// Author: Marina Suvorova (m.dudarenko@gmail.com)

#include <cmath>

#include "artm/core/exceptions.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/score/sparsity_theta.h"

namespace artm {
namespace score {

void SparsityTheta::AppendScore(
    const Item& item,
    const Batch& batch,
    const std::vector<artm::core::Token>& token_dict,
    const artm::core::PhiMatrix& p_wt,
    const artm::ProcessBatchesArgs& args,
    const std::vector<float>& theta,
    Score* score) {
  const int topic_size = p_wt.topic_size();

  std::vector<bool> topics_to_score;
  ::google::protobuf::int64 topics_to_score_size = topic_size;
  if (config_.topic_name_size() == 0) {
    topics_to_score.assign(topic_size, true);
  } else {
    topics_to_score = core::is_member(p_wt.topic_name(), config_.topic_name());
    topics_to_score_size = config_.topic_name_size();
  }

  ::google::protobuf::int64 zero_topics_count = 0;
  for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
    if ((fabs(theta[topic_index]) < config_.eps()) &&
        topics_to_score[topic_index]) {
      ++zero_topics_count;
    }
  }

  SparsityThetaScore sparsity_theta_score;
  sparsity_theta_score.set_zero_topics(zero_topics_count);
  sparsity_theta_score.set_total_topics(topics_to_score_size);
  AppendScore(sparsity_theta_score, score);
}

std::shared_ptr<Score> SparsityTheta::CreateScore() {
  return std::make_shared<SparsityThetaScore>();
}

void SparsityTheta::AppendScore(const Score& score, Score* target) {
  std::string error_message = "Unable downcast Score to SparsityThetaScore";
  const SparsityThetaScore* sparsity_theta_score = dynamic_cast<const SparsityThetaScore*>(&score);
  if (sparsity_theta_score == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(error_message));
  }

  SparsityThetaScore* sparsity_theta_target = dynamic_cast<SparsityThetaScore*>(target);
  if (sparsity_theta_target == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(error_message));
  }

  sparsity_theta_target->set_zero_topics(sparsity_theta_target->zero_topics() +
                                         sparsity_theta_score->zero_topics());
  sparsity_theta_target->set_total_topics(sparsity_theta_target->total_topics() +
                                          sparsity_theta_score->total_topics());
  sparsity_theta_target->set_value(static_cast<float>(sparsity_theta_target->zero_topics()) /
                                    sparsity_theta_target->total_topics());
}

}  // namespace score
}  // namespace artm
