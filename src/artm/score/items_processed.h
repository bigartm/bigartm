/* Copyright 2014, Additive Regularization of Topic Models.

   Author: Marina Suvorova (m.dudarenko@gmail.com)
   
   This class proceeds count the number of documents, currently
   processed by the algorithm.
   
   This score has no input parameters.
*/

#ifndef SRC_ARTM_SCORE_ITEMS_PROCESSED_H_
#define SRC_ARTM_SCORE_ITEMS_PROCESSED_H_

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
      const Item& item,
      const std::vector<artm::core::Token>& token_dict,
      const artm::core::PhiMatrix& p_wt,
      const artm::ProcessBatchesArgs& args,
      const std::vector<float>& theta,
      Score* score);

  virtual void AppendScore(
      const Batch& batch,
      Score* score);

  virtual ScoreType score_type() const { return ::artm::ScoreType_ItemsProcessed; }

 private:
  ItemsProcessedScoreConfig config_;
};

}  // namespace score
}  // namespace artm

#endif  // SRC_ARTM_SCORE_ITEMS_PROCESSED_H_
