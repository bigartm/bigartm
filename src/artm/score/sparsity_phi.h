/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Marina Suvorova (m.dudarenko@gmail.com)
   
   This class proceeds scoring of sparsity of Phi matrix.
   
   Parameters:
   - topic_name (array with topic names to score)
   - eps
   - class_id (class_id to score, empty -> DefaultClass)
   - transaction_typename (transaction typename to score, empty -> DefaultTransactionTypeName)

*/

#pragma once

#include <string>
#include <vector>

#include "artm/score_calculator_interface.h"

namespace artm {
namespace score {

class SparsityPhi : public ScoreCalculatorInterface {
 public:
  explicit SparsityPhi(const ScoreConfig& config) : ScoreCalculatorInterface(config) {
    config_ = ParseConfig<SparsityPhiScoreConfig>();
  }

  std::shared_ptr<Score> CalculateScore(const artm::core::PhiMatrix& p_wt);

  virtual bool is_cumulative() const { return false; }

  virtual ScoreType score_type() const { return ::artm::ScoreType_SparsityPhi; }

 private:
  SparsityPhiScoreConfig config_;
};

}  // namespace score
}  // namespace artm
