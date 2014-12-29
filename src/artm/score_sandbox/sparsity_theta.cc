// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/score_sandbox/sparsity_theta.h"

#include <cmath>

#include "artm/core/exceptions.h"
#include "artm/core/topic_model.h"

namespace artm {
namespace score_sandbox {

void SparsityTheta::AppendScore(
    const Item& item,
    const std::vector<artm::core::Token>& token_dict,
    const artm::core::TopicModel& topic_model,
    const std::vector<float>& theta,
    Score* score) {
  int topics_count = topic_model.topic_size();
  auto topic_name = topic_model.topic_name();
  std::vector<bool> topics_to_score;
  int topics_to_score_size = 0;

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

  int zero_topics_count = 0;
  for (int topic_index = 0; topic_index < topics_count; ++topic_index) {
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

std::string SparsityTheta::stream_name() const {
  return config_.stream_name();
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
  sparsity_theta_target->set_value(static_cast<double>(sparsity_theta_target->zero_topics()) /
                                    sparsity_theta_target->total_topics());
}

}  // namespace score_sandbox
}  // namespace artm
