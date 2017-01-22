// Copyright 2017, Additive Regularization of Topic Models.

#include "artm/score_calculator_interface.h"

#include "artm/core/dictionary.h"
#include "artm/core/phi_matrix.h"
#include "artm/core/instance.h"

namespace artm {

std::shared_ptr< ::artm::core::Dictionary> ScoreCalculatorInterface::dictionary(const std::string& dictionary_name) {
  return ::artm::core::ThreadSafeDictionaryCollection::singleton().get(dictionary_name);
}

std::shared_ptr<const ::artm::core::PhiMatrix> ScoreCalculatorInterface::GetPhiMatrix(const std::string& model_name) {
  return instance_->GetPhiMatrixSafe(model_name);
}

std::shared_ptr<Score> ScoreCalculatorInterface::CalculateScore() {
  auto phi_matrix = GetPhiMatrix(model_name());
  return CalculateScore(*phi_matrix);
}

}  // namespace artm
