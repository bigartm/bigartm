/* Copyright 2014, Additive Regularization of Topic Models.

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
   - direct_kl (true means KL(p(t) || p(t|w)), false - KL(p(t|w) || p(t)),
                default true)
*/

#ifndef SRC_ARTM_SCORE_BACKGROUND_TOKENS_PART_H_
#define SRC_ARTM_SCORE_BACKGROUND_TOKENS_PART_H_

#include <string>
#include <vector>

#include "artm/score_calculator_interface.h"
#include "artm/core/instance.h"

namespace artm {
namespace score {

class BackgroundTokensPart : public ScoreCalculatorInterface {
 public:
  explicit BackgroundTokensPart(const ScoreConfig& config) : ScoreCalculatorInterface(config) {
    config_ = ParseConfig<BackgroundTokensPartScoreConfig>();
  }

  std::shared_ptr<Score> CalculateScore(const artm::core::PhiMatrix& p_wt);

  virtual bool is_cumulative() const { return false; }

  virtual ScoreData_Type score_type() const { return ::artm::ScoreData_Type_BackgroundTokensPart; }

 private:
  BackgroundTokensPartScoreConfig config_;
};

}  // namespace score
}  // namespace artm

#endif  // SRC_ARTM_SCORE_BACKGROUND_TOKENS_PART_H_
