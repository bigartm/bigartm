// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_SCORE_CALCULATOR_INTERFACE_H_
#define SRC_ARTM_SCORE_CALCULATOR_INTERFACE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "artm/core/dictionary.h"
#include "artm/core/common.h"
#include "artm/core/phi_matrix.h"
#include "artm/core/exceptions.h"
#include "artm/messages.pb.h"

namespace artm {

typedef ::google::protobuf::Message Score;

class ScoreCalculatorInterface {
 public:
  explicit ScoreCalculatorInterface(const ScoreConfig& score_config) : score_config_(score_config) {}

  virtual ~ScoreCalculatorInterface() { }

  virtual ScoreData_Type score_type() const = 0;

  // Non-cumulative calculation (based on Phi matrix)
  virtual std::shared_ptr<Score> CalculateScore(const artm::core::PhiMatrix& p_wt) { return nullptr; }

  // Cumulative calculation (such as perplexity, or sparsity of Theta matrix)
  virtual bool is_cumulative() const { return false; }

  virtual std::shared_ptr<Score> CreateScore() { return nullptr; }

  virtual void AppendScore(const Score& score, Score* target) { return; }

  virtual void AppendScore(
      const Item& item,
      const std::vector<artm::core::Token>& token_dict_,
      const artm::core::PhiMatrix& p_wt,
      const artm::ProcessBatchesArgs& args,
      const std::vector<float>& theta,
      Score* score) { }

  virtual void AppendScore(
      const Batch& batch,
      Score* score) {}

  std::shared_ptr< ::artm::core::Dictionary> dictionary(const std::string& dictionary_name);
  void set_dictionaries(const ::artm::core::ThreadSafeDictionaryCollection* dictionaries);

  std::string model_name() const { return score_config_.model_name(); }
  std::string score_name() const { return score_config_.name(); }

  template<typename ConfigType>
  ConfigType ParseConfig() const;

 private:
  ScoreConfig score_config_;
  const ::artm::core::ThreadSafeDictionaryCollection* dictionaries_;
};

template<typename ConfigType>
ConfigType ScoreCalculatorInterface::ParseConfig() const {
  ConfigType config;
  if (!score_config_.has_config())
    return config;

  const std::string& config_blob = score_config_.config();
  if (!config.ParseFromArray(config_blob.c_str(), config_blob.length()))
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException("Unable to parse score config"));
  return config;
}

}  // namespace artm

#endif  // SRC_ARTM_SCORE_CALCULATOR_INTERFACE_H_
