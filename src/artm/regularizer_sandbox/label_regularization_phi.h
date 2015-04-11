// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#ifndef SRC_ARTM_REGULARIZER_SANDBOX_LABEL_REGULARIZATION_PHI_H_
#define SRC_ARTM_REGULARIZER_SANDBOX_LABEL_REGULARIZATION_PHI_H_

#include "artm/messages.pb.h"
#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer_sandbox {

class LabelRegularizationPhi : public RegularizerInterface {
 public:
  explicit LabelRegularizationPhi(const LabelRegularizationPhiConfig& config)
    : config_(config) {}

  virtual bool RegularizePhi(::artm::core::Regularizable* topic_model, double tau);
  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  LabelRegularizationPhiConfig config_;
};

}  // namespace regularizer_sandbox
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_SANDBOX_LABEL_REGULARIZATION_PHI_H_
