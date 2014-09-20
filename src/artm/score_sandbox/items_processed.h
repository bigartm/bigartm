// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_SCORE_SANDBOX_ITEMS_PROCESSED_H_
#define SRC_ARTM_SCORE_SANDBOX_ITEMS_PROCESSED_H_

#include <string>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/score_calculator_interface.h"

namespace artm {
namespace score_sandbox {

class ItemsProcessed : public ScoreCalculatorInterface {
 public:
  explicit ItemsProcessed(const ItemsProcessedScoreConfig& config)
    : config_(config) {}

  virtual bool is_cumulative() const { return true; }

  virtual std::string stream_name() const;

  virtual std::shared_ptr<Score> CreateScore();

  virtual void AppendScore(const Score& score, Score* target);

  virtual void AppendScore(
      const Item& item,
      const std::vector<artm::core::Token>& token_dict,
      const artm::core::TopicModel& topic_model,
      const std::vector<float>& theta,
      Score* score);

  virtual ScoreData_Type score_type() const { return ::artm::ScoreData_Type_ItemsProcessed; }

 private:
  ItemsProcessedScoreConfig config_;
};

}  // namespace score_sandbox
}  // namespace artm

#endif  // SRC_ARTM_SCORE_SANDBOX_ITEMS_PROCESSED_H_
