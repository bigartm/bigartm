/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)
   
   This class for each topic in Phi matrix counts the kernel
   characteristics --- purity, contrast and size.
   Also it can count the coherence of topics using kernel tokens.
   
   The token is kernel for topic if its p(t|w) > threshold.
   
   Parameters:
   - probability_mass_threshold (the p(t|w) threshold, default == 0.1)
   - topic_name (names of topics from which top tokens need to be extracted)
   - cooccurrence_dictionary_name (dictionary with information about
     pairwise tokens cooccurrence, strongly required)
   - transaction_typename (transaction typename to score, empty -> DefaultTransactionTypeName)
   - class_id (class_id to use, empty == DefaultClass)
   - eps

*/

#pragma once

#include <string>
#include <vector>

#include "artm/score_calculator_interface.h"
#include "artm/core/instance.h"

namespace artm {
namespace score {

class TopicKernel : public ScoreCalculatorInterface {
 public:
  explicit TopicKernel(const ScoreConfig& config) : ScoreCalculatorInterface(config) {
    config_ = ParseConfig<TopicKernelScoreConfig>();
  }

  std::shared_ptr<Score> CalculateScore(const artm::core::PhiMatrix& p_wt);

  virtual bool is_cumulative() const { return false; }

  virtual ScoreType score_type() const { return ::artm::ScoreType_TopicKernel; }

 private:
  TopicKernelScoreConfig config_;
};

}  // namespace score
}  // namespace artm
