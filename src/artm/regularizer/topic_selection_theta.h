/* Copyright 2014, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)

   This class proceeds topic selection according to Theta matrix.
   The formula of M-step is
   
   p_td \propto n_td - tau * n_td * topic_value[t],
   
   where topic_value[t] = n / (n_t * |T|) --- shuld be defined
   by user for each topic, and alpha_iter is an array of
   additional coefficients, one per document pass. If n_td is
   negative, nothing will be done.
   
   The parameters of the regularizer:
   - topic_names (the names of topics to regularize, empty == all)
   - topic_value (an array with floats with length == number of topics)
   - alpha_iter (an array of floats with length == number of
     inner iterations)

*/

#ifndef SRC_ARTM_REGULARIZER_TOPIC_SELECTION_THETA_H_
#define SRC_ARTM_REGULARIZER_TOPIC_SELECTION_THETA_H_

#include <memory>
#include <string>
#include <vector>

#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class TopicSelectionThetaAgent : public RegularizeThetaAgent {
 public:
  virtual void Apply(int item_index, int inner_iter, int topics_size, float* theta) const;

 private:
  friend class TopicSelectionTheta;

  std::vector<float> topic_weight;
  std::vector<float> alpha_weight;
  std::vector<float> topic_value;
};

class TopicSelectionTheta : public RegularizerInterface {
 public:
  explicit TopicSelectionTheta(const TopicSelectionThetaConfig& config);

  virtual std::shared_ptr<RegularizeThetaAgent>
  CreateRegularizeThetaAgent(const Batch& batch, const ProcessBatchesArgs& args, double tau);

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  TopicSelectionThetaConfig config_;
};

}  // namespace regularizer
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_TOPIC_SELECTION_THETA_H_
