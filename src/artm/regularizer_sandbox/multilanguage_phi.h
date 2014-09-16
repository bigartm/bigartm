// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_REGULARIZER_SANDBOX_MULTILANGUAGE_PHI_H_
#define SRC_ARTM_REGULARIZER_SANDBOX_MULTILANGUAGE_PHI_H_

#include "artm/messages.pb.h"
#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer_sandbox {

class MultiLanguagePhi : public RegularizerInterface {
 public:
  explicit MultiLanguagePhi(const MultiLanguagePhiConfig& config)
    : config_(config)
    , no_regularization_calls_(0) {}

  virtual bool RegularizePhi(::artm::core::Regularizable* topic_model, double tau);
  virtual bool Reconfigure(const RegularizerConfig& config);

  virtual void SerializeInternalState(RegularizerInternalState* regularizer_state);

 private:
  MultiLanguagePhiConfig config_;
  int no_regularization_calls_;
};

}  // namespace regularizer_sandbox
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_SANDBOX_MULTILANGUAGE_PHI_H_
