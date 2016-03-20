/* Copyright 2014, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)

   This class proceeds Theta matrix smoothing or sparsing.
   The formula of M-step is
   
   p_td \propto n_td + tau * alpha_iter[iter] * f(p_td) * n_td,
   
   where f is a transform function, which is p_wt multiplied on
   the derivative of function under KL-divergence, and alpha_iter
   is an array of additional coefficients, one per document pass.
   
   The parameters of the regularizer:
   - topic_names (the names of topics to regularize, empty == all)
   - transform_function (default is 1, corresponds log() under
     KL-divergence)
   - alpha_iter (an array of floats with length == number of
     inner iterations)

*/

#ifndef SRC_ARTM_REGULARIZER_SMOOTH_SPARSE_THETA_H_
#define SRC_ARTM_REGULARIZER_SMOOTH_SPARSE_THETA_H_

#include <memory>
#include <string>
#include <vector>

#include "artm/regularizer_interface.h"
#include "artm/core/transform_function.h"

namespace artm {
namespace regularizer {

class SmoothSparseThetaAgent : public RegularizeThetaAgent {
 public:
  explicit SmoothSparseThetaAgent(std::shared_ptr<artm::core::TransformFunction> func)
    : transform_function_(func) { }
  virtual void Apply(int item_index, int inner_iter, int topics_size, float* theta) const;

 private:
  friend class SmoothSparseTheta;

  std::vector<float> topic_weight;
  std::vector<float> alpha_weight;

  std::shared_ptr<artm::core::TransformFunction> transform_function_;
};

class SmoothSparseTheta : public RegularizerInterface {
 public:
  explicit SmoothSparseTheta(const SmoothSparseThetaConfig& config);

  virtual std::shared_ptr<RegularizeThetaAgent>
  CreateRegularizeThetaAgent(const Batch& batch, const ProcessBatchesArgs& args, double tau);

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  SmoothSparseThetaConfig config_;
  std::shared_ptr<artm::core::TransformFunction> transform_function_;
};

}  // namespace regularizer
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_SMOOTH_SPARSE_THETA_H_
