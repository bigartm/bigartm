// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_SCORE_TOP_TOKENS_H_
#define SRC_ARTM_SCORE_TOP_TOKENS_H_

#include <string>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/score_calculator_interface.h"

namespace artm {
namespace score {

class TopTokens : public ScoreCalculatorInterface {
 public:
  explicit TopTokens(const TopTokensScoreConfig& config)
    : config_(config) {}

  virtual std::shared_ptr<Score> CalculateScore(const artm::core::PhiMatrix& p_wt);

  virtual bool is_cumulative() const { return false; }

  virtual ScoreData_Type score_type() const { return ::artm::ScoreData_Type_TopTokens; }

 private:
  TopTokensScoreConfig config_;
};

}  // namespace score
}  // namespace artm

#endif  // SRC_ARTM_SCORE_TOP_TOKENS_H_
