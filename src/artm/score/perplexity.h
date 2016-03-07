/* Copyright 2014, Additive Regularization of Topic Models.

   Author: Marina Suvorova (m.dudarenko@gmail.com)
   
   This class proceeds scoring of perplexity.
   
   Parameters:
   - model_type (the mode of replacing zero values, default is unigram doc model) 
   - dictionary_name
   - theta_sparsity_topic_name (topic names to count Theta sparsity)
   - theta_sparsity_eps
   - class_id (array of class_ids to count perplexity, empty == all)

*/

#ifndef SRC_ARTM_SCORE_PERPLEXITY_H_
#define SRC_ARTM_SCORE_PERPLEXITY_H_

#include <string>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/score_calculator_interface.h"

namespace artm {
namespace score {

class Perplexity : public ScoreCalculatorInterface {
 public:
  explicit Perplexity(const PerplexityScoreConfig& config);

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

  virtual ScoreData_Type score_type() const { return ::artm::ScoreData_Type_Perplexity; }

 private:
  PerplexityScoreConfig config_;
};

}  // namespace score
}  // namespace artm

#endif  // SRC_ARTM_SCORE_PERPLEXITY_H_
