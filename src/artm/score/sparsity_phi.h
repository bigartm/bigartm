/* Copyright 2014, Additive Regularization of Topic Models.

   Author: Marina Suvorova (m.dudarenko@gmail.com)
   
   This class proceeds scoring of sparsity of Phi matrix.
   
   Parameters:
   - topic_name (array with topic names to score)
   - eps
   - class_id (class_id to score, default == DefaultClass)

*/

#ifndef SRC_ARTM_SCORE_SPARSITY_PHI_H_
#define SRC_ARTM_SCORE_SPARSITY_PHI_H_

#include <string>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/score_calculator_interface.h"

namespace artm {
namespace score {

class SparsityPhi : public ScoreCalculatorInterface {
 public:
  explicit SparsityPhi(const SparsityPhiScoreConfig& config)
    : config_(config) {}

  std::shared_ptr<Score> CalculateScore(const artm::core::PhiMatrix& p_wt);

  virtual bool is_cumulative() const { return false; }

  virtual ScoreData_Type score_type() const { return ::artm::ScoreData_Type_SparsityPhi; }

 private:
  SparsityPhiScoreConfig config_;
};

}  // namespace score
}  // namespace artm

#endif  // SRC_ARTM_SCORE_SPARSITY_PHI_H_
