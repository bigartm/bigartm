/* Copyright 2014, Additive Regularization of Topic Models.

   Author: Alexander Frey (sashafrey@gmail.com)
   
   This class proceeds scoring of class precision. To use it
   you need to set the 'predict_class_id' in ProcessBatchesArgs.
   In this case ProcessBatches will return the p(c|d) matrix, 
   where p(c|d)=sum_t p(c|t)*p(t|d). This score will count the
   precision of the classification, if each document has only one
   class label.
   
   Note: work on this score is in progress.
   
   This score has no input parameters.

*/

#ifndef SRC_ARTM_SCORE_CLASS_PRECISION_H_
#define SRC_ARTM_SCORE_CLASS_PRECISION_H_

#include <string>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/score_calculator_interface.h"

namespace artm {
namespace score {

class ClassPrecision : public ScoreCalculatorInterface {
 public:
  explicit ClassPrecision(const ClassPrecisionScoreConfig& config)
    : config_(config) {}

  virtual bool is_cumulative() const { return true; }

  virtual std::string stream_name() const;

  virtual std::shared_ptr<Score> CreateScore();

  virtual void AppendScore(const Score& score, Score* target);

  virtual void AppendScore(
      const Item& item,
      const std::vector<artm::core::Token>& token_dict,
      const artm::core::PhiMatrix& p_wt,
      const artm::ModelConfig& model_config,
      const std::vector<float>& theta,
      Score* score);

  virtual ScoreData_Type score_type() const { return ::artm::ScoreData_Type_ClassPrecision; }

 private:
  ClassPrecisionScoreConfig config_;
};

}  // namespace score
}  // namespace artm

#endif  // SRC_ARTM_SCORE_CLASS_PRECISION_H_
