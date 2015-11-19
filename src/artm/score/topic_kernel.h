/* Copyright 2014, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)
   
   This class for each topic in Phi matrix counts the kernel
   characteristics --- purity, contrast and size.
   Also it can count the coherency of topics using kernel tokens.
   
   The token is kernel for topic if its p(t|w) > threshold.
   
   Parameters:
   - probability_mass_threshold (the p(t|w) threshold, default == 0.1)
   - topic_name (names of topics from which top tokens need to be extracted)
   - cooccurrence_dictionary_name (dictionary with information about
     pairwise tokens cooccurrence, strongly required)
   - class_id (class_id to use, empty == DefaultClass)
   - eps

*/

#ifndef SRC_ARTM_SCORE_TOPIC_KERNEL_H_
#define SRC_ARTM_SCORE_TOPIC_KERNEL_H_

#include <string>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/score_calculator_interface.h"

namespace artm {
namespace score {

class TopicKernel : public ScoreCalculatorInterface {
 public:
  explicit TopicKernel(const TopicKernelScoreConfig& config)
    : config_(config) {}

  std::shared_ptr<Score> CalculateScore(const artm::core::PhiMatrix& p_wt);

  virtual bool is_cumulative() const { return false; }

  virtual ScoreData_Type score_type() const { return ::artm::ScoreData_Type_TopicKernel; }

 private:
  TopicKernelScoreConfig config_;
};

}  // namespace score
}  // namespace artm

#endif  // SRC_ARTM_SCORE_TOPIC_KERNEL_H_
