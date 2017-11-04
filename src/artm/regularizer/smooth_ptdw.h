/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Anya Potapenko (anya_potapenko@mail.ru)

   Work in progress, description will be provided later.
*/

#pragma once

#include <string>
#include <vector>

#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class SmoothPtdwAgent : public RegularizePtdwAgent {
 private:
  friend class SmoothPtdw;
  SmoothPtdwConfig config_;
  ProcessBatchesArgs args_;
  float tau_;

 public:
  SmoothPtdwAgent(const SmoothPtdwConfig& config, const ProcessBatchesArgs& args, float tau)
      : config_(config)
      , args_(args)
      , tau_(tau) { }

  virtual void Apply(int item_index, int inner_iter, ::artm::utility::LocalPhiMatrix<float>* ptdw) const;
};

class SmoothPtdw : public RegularizerInterface {
 public:
  explicit SmoothPtdw(const SmoothPtdwConfig& config) : config_(config) { }

  virtual std::shared_ptr<RegularizePtdwAgent>
  CreateRegularizePtdwAgent(const Batch& batch, const ProcessBatchesArgs& args, float tau);

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  SmoothPtdwConfig config_;
};

}  // namespace regularizer
}  // namespace artm
