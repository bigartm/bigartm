/* Copyright 2014, Additive Regularization of Topic Models.

   Author: Bulatov Victor

   The original formula of M-step is
   
   // p_td \propto n_td + tau * sum_v w_dv * p_tv,
   // the idea is to mix topical distribution of documents if they are linked together in some way
   // (e.g. are citing each other)

   // this operation is difficult to parallelize, 
   // so this implementation uses this modified formula instead:
   // p_td \propto n_td + tau * sum_v w_dv * phi_vt


*/

#ifndef SRC_ARTM_REGULARIZER_ITOPIC_THETA_H_
#define SRC_ARTM_REGULARIZER_ITOPIC_THETA_H_

#include <memory>
#include <string>
#include <vector>

#include "artm/regularizer_interface.h"
#include "artm/core/instance.h"
#include "artm/core/token.h"

namespace artm {
namespace regularizer {

class iTopicThetaAgent : public RegularizeThetaAgent {
 public:
  virtual void Apply(int item_index, int inner_iter, int topics_size, float* theta) const;
  explicit iTopicThetaAgent(const Batch& batch, const ::artm::core::Instance* instance_) : 
      mybatch(batch), myinstance_(instance_) {}

 private:
  friend class iTopicTheta;
  const Batch& mybatch;
  const ::artm::core::Instance* myinstance_;
  iTopicThetaConfig config_;
  float tau_;
};

class iTopicTheta : public RegularizerInterface {
 public:
  explicit iTopicTheta(const iTopicThetaConfig& config) : config_(config) { }
  virtual std::shared_ptr<RegularizeThetaAgent>
  CreateRegularizeThetaAgent(const Batch& batch, const ProcessBatchesArgs& args, double tau);

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  iTopicThetaConfig config_;
};

}  // namespace regularizer
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_TOPIC_SELECTION_THETA_H_
