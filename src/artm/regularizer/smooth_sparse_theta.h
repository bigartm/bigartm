/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)

   This class proceeds Theta matrix smoothing or sparsing.
   The formula of M-step is
   
   p_td \propto n_td + tau * item_topic_multiplier[d][t] * alpha_iter[iter] * f(p_td) * n_td,
   
   where f is a transform function, which is p_wt multiplied on
   the derivative of function under KL-divergence, and alpha_iter
   is an array of additional coefficients, one per document pass.
   
   The parameters of the regularizer:
   - topic_names (the names of topics to regularize, empty == all)
   - transform_function (default is 1, corresponds log() under
     KL-divergence)
   - alpha_iter (an array of floats with length == number of
     inner iterations)
   - item_title (an array of string. If not empty => only items with
     titles from this array will be regularized)
   - item_topic_multiplier (an array of arrays of floats. Should have
     length 1 or equal to length of item_title)

*/

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "artm/regularizer_interface.h"
#include "artm/core/transform_function.h"

namespace artm {
namespace regularizer {

typedef std::unordered_map<std::string, std::vector<float> > ItemTopicMultiplier;

class SmoothSparseThetaAgent : public RegularizeThetaAgent {
 public:
  explicit SmoothSparseThetaAgent(const Batch& batch,
                                  std::shared_ptr<artm::core::TransformFunction> func,
                                  std::shared_ptr<ItemTopicMultiplier> item_topic_multiplier,
                                  std::shared_ptr<std::vector<float> > universal_topic_multiplier)
    : batch_(batch)
    , transform_function_(func)
    , item_topic_multiplier_(item_topic_multiplier)
    , universal_topic_multiplier_(universal_topic_multiplier) { }

  virtual void Apply(int item_index, int inner_iter, int topics_size, const float* n_td, float* r_td) const;

 private:
  friend class SmoothSparseTheta;

  const Batch& batch_;
  std::vector<float> topic_weight;
  std::vector<float> alpha_weight;

  std::shared_ptr<artm::core::TransformFunction> transform_function_;
  std::shared_ptr<ItemTopicMultiplier> item_topic_multiplier_;
  std::shared_ptr<std::vector<float> > universal_topic_multiplier_;
};

class SmoothSparseTheta : public RegularizerInterface {
 public:
  explicit SmoothSparseTheta(const SmoothSparseThetaConfig& config);

  virtual std::shared_ptr<RegularizeThetaAgent>
  CreateRegularizeThetaAgent(const Batch& batch, const ProcessBatchesArgs& args, float tau);

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();

  virtual bool Reconfigure(const RegularizerConfig& config);

  void ReconfigureImpl();

 private:
  SmoothSparseThetaConfig config_;
  std::shared_ptr<artm::core::TransformFunction> transform_function_;
  std::shared_ptr<ItemTopicMultiplier> item_topic_multiplier_;
  std::shared_ptr<std::vector<float> > universal_topic_multiplier_;
};

}  // namespace regularizer
}  // namespace artm
