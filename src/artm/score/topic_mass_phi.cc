// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/exceptions.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/score/topic_mass_phi.h"

namespace artm {
namespace score {

std::shared_ptr<Score> TopicMassPhi::CalculateScore(const artm::core::PhiMatrix& p_wt) {
  // this score counts the n_t counters of 'topic_name'
  // part in the n_t counters of all topics
  int topic_size = p_wt.topic_size();
  int token_size = p_wt.token_size();

  // parameters preparation
  std::vector<bool> topics_to_score;
  if (config_.topic_name_size() == 0)
    topics_to_score.assign(topic_size, true);
  else
    topics_to_score = core::is_member(p_wt.topic_name(), config_.topic_name());

  ::artm::core::ClassId class_id = ::artm::core::DefaultClass;
  if (config_.has_class_id())
    class_id = config_.class_id();

  std::vector<double> topic_mass;
  topic_mass.assign(topic_size, 0.0);
  double denominator = 0.0;
  double numerator = 0.0;

  for (int token_index = 0; token_index < token_size; token_index++) {
    if (p_wt.token(token_index).class_id == class_id) {
      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
        double value = p_wt.get(token_index, topic_index);
        denominator += value;
        topic_mass[topic_index] += value;

        if (topics_to_score[topic_index])
          numerator += value;
      }
    }
  }

  TopicMassPhiScore* topic_mass_score = new TopicMassPhiScore();
  std::shared_ptr<Score> retval(topic_mass_score);

  double value = 0.0;
  if (denominator > config_.eps())
    value = static_cast<double>(numerator / denominator);
  topic_mass_score->set_value(value);

  for (auto& name : p_wt.topic_name())
      topic_mass_score->add_topic_name(name);

  for (double elem : topic_mass)
    topic_mass_score->add_topic_mass(elem);

  return retval;
}

}  // namespace score
}  // namespace artm
