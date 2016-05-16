/* Copyright 2014, Additive Regularization of Topic Models.

   Author: me

   This class proceeds
   The formula of M-step is
   
   // p_td \propto n_td - tau * n_td * topic_value[t],
   
   // where topic_value[t] = n / (n_t * |T|) --- shuld be defined
   // by user for each topic, and alpha_iter is an array of
   // additional coefficients, one per document pass. If n_td is
   // negative, nothing will be done.
   
   The parameters of the regularizer:

*/

#ifndef SRC_ARTM_REGULARIZER_ITOPIC_THETA_H_
#define SRC_ARTM_REGULARIZER_ITOPIC_THETA_H_

#include <memory>
#include <string>
#include <vector>

#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class iTopicThetaAgent : public RegularizeThetaAgent {
 public:
  virtual void Apply(int item_index, int inner_iter, int topics_size, float* theta) const;

 private:
  friend class iTopicTheta;

};

class iTopicTheta : public RegularizerInterface {
 public:
  explicit iTopicTheta(const iTopicThetaConfig& config);

  virtual std::shared_ptr<RegularizeThetaAgent>
  CreateRegularizeThetaAgent(const Batch& batch, const ProcessBatchesArgs& args, double tau);

  virtual google::protobuf::RepeatedPtrField<std::string> class_name();

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  iTopicThetaConfig config_;
};

}  // namespace regularizer
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_TOPIC_SELECTION_THETA_H_
