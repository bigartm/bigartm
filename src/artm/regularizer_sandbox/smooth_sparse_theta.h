// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#ifndef SRC_ARTM_REGULARIZER_SANDBOX_SMOOTH_SPARSE_THETA_H_
#define SRC_ARTM_REGULARIZER_SANDBOX_SMOOTH_SPARSE_THETA_H_

#include <vector>

#include "artm/messages.pb.h"
#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer_sandbox {

class SmoothSparseTheta : public RegularizerInterface {
 public:
  explicit SmoothSparseTheta(const SmoothSparseThetaConfig& config)
    : config_(config) {}

  virtual bool RegularizeTheta(const Item& item,
                               std::vector<float>* n_dt,
                               int topic_size,
                               int inner_iter,
                               double tau);
  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  SmoothSparseThetaConfig config_;
};

}  // namespace regularizer_sandbox
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_SANDBOX_SMOOTH_SPARSE_THETA_H_
