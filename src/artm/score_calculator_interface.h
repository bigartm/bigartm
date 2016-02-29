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
#include "artm/messages.pb.h"

namespace artm {

typedef ::google::protobuf::Message Score;

class ScoreCalculatorInterface {
 public:
  ScoreCalculatorInterface() {}
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

 private:
  const ::artm::core::ThreadSafeDictionaryCollection* dictionaries_;
};
}  // namespace artm

#endif  // SRC_ARTM_SCORE_CALCULATOR_INTERFACE_H_
