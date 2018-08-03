/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)
   
   This class count the n_t values for each topic in Phi matrix.
   
   Parameters:
   - topic_name (names of topics to compute n_t)
   - transaction_typename (transaction typename to score, empty -> DefaultTransactionTypeName)
   - class_id (class_id to use, empty == all modalities)
   - eps

*/

#pragma once

#include <string>
#include <vector>

#include "artm/score_calculator_interface.h"

namespace artm {
namespace score {

class TopicMassPhi : public ScoreCalculatorInterface {
 public:
  explicit TopicMassPhi(const ScoreConfig& config) : ScoreCalculatorInterface(config) {
    config_ = ParseConfig<TopicMassPhiScoreConfig>();
  }

  std::shared_ptr<Score> CalculateScore(const artm::core::PhiMatrix& p_wt);

  virtual bool is_cumulative() const { return false; }

  virtual ScoreType score_type() const { return ::artm::ScoreType_TopicMassPhi; }

 private:
  TopicMassPhiScoreConfig config_;
};

}  // namespace score
}  // namespace artm
