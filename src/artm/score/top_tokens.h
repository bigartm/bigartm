/* Copyright 2014, Additive Regularization of Topic Models.

   Authors: Marina Suvorova (m.dudarenko@gmail.com)
            Murat Apishev (great-mel@yandex.ru)
   
   This class returns the most probable tokens in each topic in Phi matrix.
   Also it can count the coherency of topics using top tokens.
   
   Parameters:
   - num_tokens (the number of top tokens to extract from each topic) 
   - topic_name (names of topics from which top tokens need to be extracted)
   - cooccurrence_dictionary_name (dictionary with information about
     pairwise tokens cooccurrence, strongly required)
   - class_id (class_id to use, empty == DefaultClass)

*/

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
