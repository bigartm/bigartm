/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)
   
   This class computes KL(p(t) || p(t|w)) (or via versa)
   for each token and counts the part of tokens that have
   this value greater than delta.

   Parameters:
   - delta_threshold (required value to mark token as background)
   - save_tokens (return background tokens in 'tokens' field or
                  not, default = true)
   - dictionary_name (dictionary contains original 'value' field
                      that will be used as p(w) instead of
                      calculating it)
   - transaction_typename (transaction typename to score, empty -> DefaultTransactionTypeName)
   - direct_kl (true means KL(p(t) || p(t|w)), false - KL(p(t|w) || p(t)),
                default true)
*/

#pragma once

#include <string>
#include <vector>

#include "artm/score_calculator_interface.h"
#include "artm/core/instance.h"

namespace artm {
namespace score {

class BackgroundTokensRatio : public ScoreCalculatorInterface {
 public:
  explicit BackgroundTokensRatio(const ScoreConfig& config) : ScoreCalculatorInterface(config) {
    config_ = ParseConfig<BackgroundTokensRatioScoreConfig>();
  }

  std::shared_ptr<Score> CalculateScore(const artm::core::PhiMatrix& p_wt);

  virtual bool is_cumulative() const { return false; }

  virtual ScoreType score_type() const { return ::artm::ScoreType_BackgroundTokensRatio; }

 private:
  BackgroundTokensRatioScoreConfig config_;
};

}  // namespace score
}  // namespace artm
