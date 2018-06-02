/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Marina Suvorova (m.dudarenko@gmail.com)
   
   This class proceeds scoring of perplexity.
   
   Parameters:
   - model_type (the mode of replacing zero values, default is unigram doc model) 
   - dictionary_name
   - theta_sparsity_topic_name (topic names to count Theta sparsity)
   - theta_sparsity_eps
   - class_ids (class_ids to score within each transaction type, empty == all)
   - transaction_typenames (transaction typenames to regularize, empty == all)

*/

#pragma once

#include <string>
#include <vector>

#include "artm/score_calculator_interface.h"

namespace artm {
namespace score {

class Perplexity : public ScoreCalculatorInterface {
 public:
  explicit Perplexity(const ScoreConfig& config);

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

  virtual ScoreType score_type() const { return ::artm::ScoreType_Perplexity; }

 private:
  PerplexityScoreConfig config_;
};

}  // namespace score
}  // namespace artm
