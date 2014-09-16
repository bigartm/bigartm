// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#ifndef SRC_ARTM_REGULARIZER_SANDBOX_DIRICHLET_THETA_H_
#define SRC_ARTM_REGULARIZER_SANDBOX_DIRICHLET_THETA_H_

#include <vector>

#include "artm/messages.pb.h"
#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer_sandbox {

class DirichletTheta : public RegularizerInterface {
 public:
  explicit DirichletTheta(const DirichletThetaConfig& config)
    : config_(config) {}

  virtual bool RegularizeTheta(const Item& item,
                               std::vector<float>* n_dt,
                               int topic_size,
                               int inner_iter,
                               double tau);
  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  DirichletThetaConfig config_;
};

}  // namespace regularizer_sandbox
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_SANDBOX_DIRICHLET_THETA_H_
