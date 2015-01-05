// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#ifndef SRC_ARTM_REGULARIZER_SANDBOX_SMOOTH_SPARSE_THETA_H_
#define SRC_ARTM_REGULARIZER_SANDBOX_SMOOTH_SPARSE_THETA_H_

#include <string>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer_sandbox {

class SmoothSparseTheta : public RegularizerInterface {
 public:
  explicit SmoothSparseTheta(const SmoothSparseThetaConfig& config)
    : config_(config) {}

  virtual bool RegularizeTheta(const Batch& batch,
                               const ModelConfig& model_config,
                               int inner_iter,
                               double tau,
                               ::artm::utility::DenseMatrix<float>* theta);
  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  SmoothSparseThetaConfig config_;
};

}  // namespace regularizer_sandbox
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_SANDBOX_SMOOTH_SPARSE_THETA_H_
