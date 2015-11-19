/* Copyright 2014, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)
   
   This class count the n_t values for each topic in Phi matrix.
   
   Parameters:
   - topic_name (names of topics from which top tokens need to be extracted)
   - class_id (class_id to use, empty == DefaultClass)
   - eps

*/

#ifndef SRC_ARTM_SCORE_TOPIC_MASS_PHI_H_
#define SRC_ARTM_SCORE_TOPIC_MASS_PHI_H_

#include <string>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/score_calculator_interface.h"

namespace artm {
namespace score {

class TopicMassPhi : public ScoreCalculatorInterface {
 public:
  explicit TopicMassPhi(const TopicMassPhiScoreConfig& config)
    : config_(config) {}

  std::shared_ptr<Score> CalculateScore(const artm::core::PhiMatrix& p_wt);

  virtual bool is_cumulative() const { return false; }

  virtual ScoreData_Type score_type() const { return ::artm::ScoreData_Type_TopicMassPhi; }

 private:
  TopicMassPhiScoreConfig config_;
};

}  // namespace score
}  // namespace artm

#endif  // SRC_ARTM_SCORE_TOPIC_MASS_PHI_H_
