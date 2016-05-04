/* Copyright 2016, Additive Regularization of Topic Models.

   Author: Alexander Frey (sashafrey@gmail.com)
   
   This class calculates peak memory usage of the process.
   
   This score has no input parameters.

*/

#ifndef SRC_ARTM_SCORE_PEAK_MEMORY_H_
#define SRC_ARTM_SCORE_PEAK_MEMORY_H_

#include "artm/score_calculator_interface.h"

namespace artm {
namespace score {

class PeakMemory : public ScoreCalculatorInterface {
 public:
  explicit PeakMemory(const ScoreConfig& config) : ScoreCalculatorInterface(config) {}

  virtual bool is_cumulative() const { return false; }

  virtual std::shared_ptr<Score> CalculateScore(const artm::core::PhiMatrix& p_wt);

  virtual ScoreType score_type() const { return ::artm::ScoreType_PeakMemory; }
};

}  // namespace score
}  // namespace artm

#endif  // SRC_ARTM_SCORE_PEAK_MEMORY_H_
