/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Marina Suvorova (m.dudarenko@gmail.com)
   
   This class proceeds count the number of documents, currently
   processed by the algorithm.
   
   This score has no input parameters.
*/

#pragma once

#include <string>
#include <vector>

#include "artm/score_calculator_interface.h"

namespace artm {
namespace score {

class ItemsProcessed : public ScoreCalculatorInterface {
 public:
  explicit ItemsProcessed(const ScoreConfig& config) : ScoreCalculatorInterface(config) {
    config_ = ParseConfig<ItemsProcessedScoreConfig>();
  }

  virtual bool is_cumulative() const { return true; }

  virtual std::shared_ptr<Score> CreateScore();

  virtual void AppendScore(const Score& score, Score* target);

  virtual void AppendScore(
      const Batch& batch,
      const artm::core::PhiMatrix& p_wt,
      const artm::ProcessBatchesArgs& args,
      Score* score);

  virtual ScoreType score_type() const { return ::artm::ScoreType_ItemsProcessed; }

 private:
  ItemsProcessedScoreConfig config_;
};

}  // namespace score
}  // namespace artm
