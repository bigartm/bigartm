/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Marina Suvorova (m.dudarenko@gmail.com)
   
   This class can provide you the part of Theta matrix
   according to your request.
   
   Parameters:
   - item_id (array with documents ids to extract)
   - item_count (number of first documents to extract)

*/

#pragma once

#include <string>
#include <vector>

#include "artm/score_calculator_interface.h"

namespace artm {
namespace score {

class ThetaSnippet : public ScoreCalculatorInterface {
 public:
  explicit ThetaSnippet(const ScoreConfig& config) : ScoreCalculatorInterface(config) {
    config_ = ParseConfig<ThetaSnippetScoreConfig>();
  }

  virtual bool is_cumulative() const { return true; }

  virtual std::shared_ptr<Score> CreateScore();

  virtual void AppendScore(const Score& score, Score* target);

  virtual void AppendScore(
      const Item& item,
      const Batch& batch,
      const std::vector<artm::core::Token>& token_dict,
      const artm::core::PhiMatrix& p_wt,
      const artm::ProcessBatchesArgs& args,
      const std::vector<float>& theta,
      Score* score);

  virtual ScoreType score_type() const { return ::artm::ScoreType_ThetaSnippet; }

 private:
  ThetaSnippetScoreConfig config_;
};

}  // namespace score
}  // namespace artm
