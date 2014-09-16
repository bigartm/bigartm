// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/score_sandbox/sparsity_theta.h"

#include <math.h>

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
  int topics_size = topic_model.topic_size();
  int topics_to_score_size = 0;

  ::artm::BoolArray topics_to_score;
  bool has_correct_vector = false;
  if (config_.has_topics_to_score()) {
    if (config_.topics_to_score().value_size() != topics_size) {
      LOG(INFO) << "Score Sparsity Theta: len(topics_to_score) must be equal to" <<
        "len(topics_size). All topics will be used in scoring!\n";
    } else {
      has_correct_vector = true;
      topics_to_score.CopyFrom(config_.topics_to_score());
      for (int i = 0; i < topics_size; ++i) {
        if (topics_to_score.value(i)) topics_to_score_size++;
      }
    }
  }

  if (!has_correct_vector) {
    for (int topic_id = 0; topic_id < topics_size; ++topic_id) {
      topics_to_score.add_value(true);
    }
    topics_to_score_size = topics_size;
  }


  int zero_topics_count = 0;
  for (int topic_index = 0; topic_index < topics_size; ++topic_index) {
    if ((fabs(theta[topic_index]) < config_.eps()) &&
        topics_to_score.value(topic_index)) {
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
