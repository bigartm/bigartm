// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class MultiLanguagePhi : public RegularizerInterface {
 public:
  explicit MultiLanguagePhi(const MultiLanguagePhiConfig& config)
    : config_(config)
    , no_regularization_calls_(0) { }

  virtual bool RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                             const ::artm::core::PhiMatrix& n_wt,
                             ::artm::core::PhiMatrix* result);

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  MultiLanguagePhiConfig config_;
  int no_regularization_calls_;
};

}  // namespace regularizer
}  // namespace artm
