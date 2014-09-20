// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/score_sandbox/sparsity_phi.h"

#include <math.h>

#include "artm/core/exceptions.h"
#include "artm/core/topic_model.h"

namespace artm {
namespace score_sandbox {

std::shared_ptr<Score> SparsityPhi::CalculateScore(const artm::core::TopicModel& topic_model) {
  int topics_size = topic_model.topic_size();
  int tokens_size = topic_model.token_size();
  int zero_tokens_count = 0;
  int topics_to_score_size = 0;

  ::artm::BoolArray topics_to_score;
  bool has_correct_vector = false;
  if (config_.has_topics_to_score()) {
    if (config_.topics_to_score().value_size() != topics_size) {
      LOG(INFO) << "Score Sparsity Phi: len(topics_to_score) must be equal to" <<
        "len(topics_size). All topics will be used in scoring.";
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

  for (int token_index = 0; token_index < tokens_size; token_index++) {
    ::artm::core::TopicWeightIterator topic_iter = topic_model.GetTopicWeightIterator(token_index);
    while (topic_iter.NextTopic() < topics_size) {
      if ((fabs(topic_iter.Weight()) < config_.eps() ||
          fabs(topic_iter.NotNormalizedWeight()) < config_.eps()) &&
          topics_to_score.value(topic_iter.TopicIndex())) {
        ++zero_tokens_count;
      }
    }
  }

  SparsityPhiScore* sparsity_phi_score = new SparsityPhiScore();
  std::shared_ptr<Score> retval(sparsity_phi_score);

  sparsity_phi_score->set_zero_tokens(zero_tokens_count);
  sparsity_phi_score->set_total_tokens(tokens_size * topics_to_score_size);
  sparsity_phi_score->set_value(static_cast<double>(sparsity_phi_score->zero_tokens()) /
                                static_cast<double>(sparsity_phi_score->total_tokens()));

  return retval;
}

}  // namespace score_sandbox
}  // namespace artm
