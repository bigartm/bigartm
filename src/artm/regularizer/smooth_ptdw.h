/* Copyright 2015, Additive Regularization of Topic Models.

   Author: Anya Potapenko (anya_potapenko@mail.ru)

   Work in progress, description will be provided later.
*/

#ifndef SRC_ARTM_REGULARIZER_SMOOTH_PTDW_H_
#define SRC_ARTM_REGULARIZER_SMOOTH_PTDW_H_

#include <string>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class SmoothPtdwAgent : public RegularizePtdwAgent {
 private:
  friend class SmoothPtdw;
  SmoothPtdwConfig config_;
  ModelConfig model_config_;
  double tau_;

 public:
  SmoothPtdwAgent(const SmoothPtdwConfig& config, const ModelConfig& model_config, double tau)
      : config_(config), model_config_(model_config), tau_(tau) {}

  virtual void Apply(int item_index, int inner_iter, ::artm::utility::DenseMatrix<float>* ptdw) const;
};

class SmoothPtdw : public RegularizerInterface {
 public:
  explicit SmoothPtdw(const SmoothPtdwConfig& config)
    : config_(config) {}

  virtual std::shared_ptr<RegularizePtdwAgent>
  CreateRegularizePtdwAgent(const Batch& batch, const ModelConfig& model_config, double tau);

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  SmoothPtdwConfig config_;
};

}  // namespace regularizer
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_SMOOTH_PTDW_H_
