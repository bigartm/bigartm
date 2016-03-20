// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_REGULARIZER_MULTILANGUAGE_PHI_H_
#define SRC_ARTM_REGULARIZER_MULTILANGUAGE_PHI_H_

#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class MultiLanguagePhi : public RegularizerInterface {
 public:
  explicit MultiLanguagePhi(const MultiLanguagePhiConfig& config)
    : config_(config)
    , no_regularization_calls_(0) {}

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

#endif  // SRC_ARTM_REGULARIZER_MULTILANGUAGE_PHI_H_
