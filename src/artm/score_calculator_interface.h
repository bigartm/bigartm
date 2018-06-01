// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "artm/core/common.h"
#include "artm/core/dictionary.h"
#include "artm/core/phi_matrix.h"
#include "artm/core/exceptions.h"
#include "artm/core/token.h"
#include "artm/artm_export.h"
#include "artm/messages.pb.h"

namespace artm {

typedef ::google::protobuf::Message Score;

namespace core {
class Instance;
}  // namespace core

// ScoreCalculatorInterface is the base class for all score calculators in BigARTM.
// See any class in 'src/score' folder for an example of how to implement new score.
// Keep in mind that scres can be either cumulative (theta-scores) or non-cumulative (phi-scores).
// ScoreCalculatorInterface is, unfortunately, a base class for both of them.
// Hopefully we will refactor this at some point in the future.
class ScoreCalculatorInterface {
 public:
  explicit ScoreCalculatorInterface(const ScoreConfig& score_config)
      : score_config_(score_config)
      , dictionaries_(nullptr)
      , instance_(nullptr) { }

  virtual ~ScoreCalculatorInterface() { }

  virtual ScoreType score_type() const = 0;

  // Non-cumulative calculation (based on Phi matrix)
  virtual std::shared_ptr<Score> CalculateScore(const artm::core::PhiMatrix& p_wt) { return nullptr; }
  virtual std::shared_ptr<Score> CalculateScore();

  // Cumulative calculation (such as perplexity, or sparsity of Theta matrix)
  virtual bool is_cumulative() const { return false; }

  virtual std::shared_ptr<Score> CreateScore() { return nullptr; }

  virtual void AppendScore(const Score& score, Score* target) { return; }

  virtual void AppendScore(
      const Item& item,
      const Batch& batch,
      const std::vector<artm::core::Token>& token_dict_,
      const artm::core::PhiMatrix& p_wt,
      const artm::ProcessBatchesArgs& args,
      const std::vector<float>& theta,
      Score* score) { }

  virtual void AppendScore(
      const Batch& batch,
      const artm::core::PhiMatrix& p_wt,
      const artm::ProcessBatchesArgs& args,
      Score* score) { }

  std::shared_ptr< ::artm::core::Dictionary> dictionary(const std::string& dictionary_name);
  std::shared_ptr<const ::artm::core::PhiMatrix> GetPhiMatrix(const std::string& model_name);

  std::string model_name() const { return score_config_.model_name(); }
  std::string score_name() const { return score_config_.name(); }
  void set_instance(::artm::core::Instance* instance) { instance_ = instance; }

  template<typename ConfigType>
  ConfigType ParseConfig() const;

 private:
  ScoreConfig score_config_;
  const ::artm::core::ThreadSafeDictionaryCollection* dictionaries_;

 protected:
  ::artm::core::Instance* instance_;
};

template<typename ConfigType>
ConfigType ScoreCalculatorInterface::ParseConfig() const {
  ConfigType config;
  if (!score_config_.has_config()) {
    return config;
  }

  const std::string& config_blob = score_config_.config();
  if (!config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException("Unable to parse score config"));
  }
  return config;
}

}  // namespace artm
