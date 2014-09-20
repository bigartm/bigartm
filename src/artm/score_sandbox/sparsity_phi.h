// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_SCORE_SANDBOX_SPARSITY_PHI_H_
#define SRC_ARTM_SCORE_SANDBOX_SPARSITY_PHI_H_

#include <string>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/score_calculator_interface.h"

namespace artm {
namespace score_sandbox {

class SparsityPhi : public ScoreCalculatorInterface {
 public:
  explicit SparsityPhi(const SparsityPhiScoreConfig& config)
    : config_(config) {}

  std::shared_ptr<Score> CalculateScore(const artm::core::TopicModel& topic_model);

  virtual bool is_cumulative() const { return false; }

  virtual ScoreData_Type score_type() const { return ::artm::ScoreData_Type_SparsityPhi; }

 private:
  SparsityPhiScoreConfig config_;
};

}  // namespace score_sandbox
}  // namespace artm

#endif  // SRC_ARTM_SCORE_SANDBOX_SPARSITY_PHI_H_
