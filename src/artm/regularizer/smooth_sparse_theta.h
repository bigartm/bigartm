// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#ifndef SRC_ARTM_REGULARIZER_SMOOTH_SPARSE_THETA_H_
#define SRC_ARTM_REGULARIZER_SMOOTH_SPARSE_THETA_H_

#include <string>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class SmoothSparseThetaAgent : public RegularizeThetaAgent {
 private:
  friend class SmoothSparseTheta;
  std::vector<float> topic_weight;
  std::vector<float> alpha_weight;
 public:
  virtual void Apply(int item_index, int inner_iter, int topics_size, float* theta) const;
};

class SmoothSparseTheta : public RegularizerInterface {
 public:
  explicit SmoothSparseTheta(const SmoothSparseThetaConfig& config)
    : config_(config) {}

  virtual std::shared_ptr<RegularizeThetaAgent>
  CreateRegularizeThetaAgent(const Batch& batch, const ModelConfig& model_config, double tau);

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  SmoothSparseThetaConfig config_;
};

}  // namespace regularizer
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_SMOOTH_SPARSE_THETA_H_
