// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#ifndef SRC_ARTM_REGULARIZER_SANDBOX_SMOOTH_SPARSE_PHI_H_
#define SRC_ARTM_REGULARIZER_SANDBOX_SMOOTH_SPARSE_PHI_H_

#include "artm/messages.pb.h"
#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer_sandbox {

class SmoothSparsePhi : public RegularizerInterface {
 public:
  explicit SmoothSparsePhi(const SmoothSparsePhiConfig& config)
    : config_(config) {}

  virtual bool RegularizePhi(::artm::core::Regularizable* topic_model, double tau);
  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  SmoothSparsePhiConfig config_;
};

}  // namespace regularizer_sandbox
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_SANDBOX_SMOOTH_SPARSE_PHI_H_
