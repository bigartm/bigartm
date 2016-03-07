/* Copyright 2014, Additive Regularization of Topic Models.

   Author: Marina Suvorova (m.dudarenko@gmail.com)
   
   This class proceeds scoring of sparsity of Theta matrix.
   
   Parameters:
   - topic_name (array with topic names to score)
   - eps

*/

#ifndef SRC_ARTM_SCORE_SPARSITY_THETA_H_
#define SRC_ARTM_SCORE_SPARSITY_THETA_H_

#include <string>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/score_calculator_interface.h"

namespace artm {
namespace score {

class SparsityTheta : public ScoreCalculatorInterface {
 public:
  explicit SparsityTheta(const SparsityThetaScoreConfig& config)
    : config_(config) {}

  virtual bool is_cumulative() const { return true; }

  virtual std::shared_ptr<Score> CreateScore();

  virtual void AppendScore(const Score& score, Score* target);

  virtual void AppendScore(
      const Item& item,
      const std::vector<artm::core::Token>& token_dict,
      const artm::core::PhiMatrix& p_wt,
      const artm::ProcessBatchesArgs& args,
      const std::vector<float>& theta,
      Score* score);

  virtual ScoreData_Type score_type() const { return ::artm::ScoreData_Type_SparsityTheta; }

 private:
  SparsityThetaScoreConfig config_;
};

}  // namespace score
}  // namespace artm

#endif  // SRC_ARTM_SCORE_SPARSITY_THETA_H_
