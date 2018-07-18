// Copyright 2017, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include "artm/core/exceptions.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/score/topic_mass_phi.h"

namespace artm {
namespace score {

std::shared_ptr<Score> TopicMassPhi::CalculateScore(const artm::core::PhiMatrix& p_wt) {
  // parameters preparation
  const int topic_size = p_wt.topic_size();
  const int token_size = p_wt.token_size();

  std::vector<bool> topics_to_score;
  int topics_to_score_size = topic_size;
  if (config_.topic_name_size() == 0) {
    topics_to_score.assign(topic_size, true);
  } else {
    topics_to_score = core::is_member(p_wt.topic_name(), config_.topic_name());
    topics_to_score_size = config_.topic_name_size();
  }

  bool use_all_classes = false;
  if (config_.class_id_size() == 0) {
    use_all_classes = true;
  }

  std::vector<float> topic_mass;
  topic_mass.assign(topics_to_score_size, 0.0f);
  double denominator = 0.0;
  double numerator = 0.0;

  for (int token_index = 0; token_index < token_size; token_index++) {
    const auto& token = p_wt.token(token_index);
    if ((!use_all_classes && !core::is_member(token.class_id, config_.class_id()))) {
      continue;
    }

    int real_topic_index = 0;
    for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
      float value = p_wt.get(token_index, topic_index);
      denominator += value;

      if (topics_to_score[topic_index]) {
        numerator += value;
        topic_mass[real_topic_index++] += value;
      }
    }
  }

  TopicMassPhiScore* topic_mass_score = new TopicMassPhiScore();
  std::shared_ptr<Score> retval(topic_mass_score);

  float value = 0.0f;
  if (denominator > config_.eps()) {
    value = static_cast<float>(numerator / denominator);
  }
  topic_mass_score->set_value(value);

  for (int i = 0; i < topic_size; ++i) {
    if (topics_to_score[i]) {
      topic_mass_score->add_topic_name(p_wt.topic_name(i));
    }
  }

  for (const auto& elem : topic_mass) {
    // don't check denominator value: if it's near zero the 'value' will show it
    topic_mass_score->add_topic_mass(elem);
    topic_mass_score->add_topic_ratio(elem / denominator);
  }

  return retval;
}

}  // namespace score
}  // namespace artm
