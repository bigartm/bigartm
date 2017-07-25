/* Copyright 2017, Additive Regularization of Topic Models.

Author: Nadia Chirkova (nadiinchi@gmail.com)
Based on code of Murat Apishev (great-mel@yandex.ru)

This regularizer improves structure of topic hierarchy and
affects psi matrix that contains p(topic|supertopic) values.

Shortly about hierarchy construction:
we build hierarchy top down, level by level, each level is a single topic model.
Suppose you have build few levels and want to build new level.
Last built level is called parent level; its topic are called supertopics.
We create extra batch containig parent level phi columns as documents.
Corresponding to this extra batch theta matrix contains p(topic|supertopic) values
and is called Psi.

Regularizer formula (remember d is not a document but is a supertopic!):
p_td \propto n_td - tau * (1 / count(supertopics) - p(supertopic|topic)),

where count(supertopics) equals |T| of parent level,
p(supertopic|topic) = p(topic|supertopic) * p(supertopic) / p(topic).
If n_td is negative, nothing will be done.

The parameters of the regularizer:
- topic_names (the names of topics to regularize, empty == all)
- alpha_iter (an array of additional coefficients, one per document pass,
an array of floats with length == number of inner iterations,
if not passed 1.0 is used as value)
- parent_topic_proportion (an array of p(supertopic) values,
an array of floats with length == count(supertopics),
if not passed 1.0 is used as value)

!Note!
* parent_topic_proportion is optional argument. But if you pass it,
remember to take these values into account when computing p(topic)
in formula (*)!
*/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class HierarchySparsingThetaAgent : public RegularizeThetaAgent {
 public:
  virtual void Apply(int item_index, int inner_iter, int topics_size, const float* n_td, float* r_td) const;
  virtual void Apply(int inner_iter,
    const ::artm::utility::LocalThetaMatrix<float>& n_td,
    ::artm::utility::LocalThetaMatrix<float>* r_td) const;

 private:
  friend class HierarchySparsingTheta;

  std::vector<float> topic_weight;              // tau for every topic
  std::vector<float> alpha_weight;              // iteration coef
  std::vector<float> parent_topic_proportion;   // p(supertopic)
  float prior_parent_topic_probability;         // 1.0 / count(supertopics)
  bool regularization_on = true;                // true if current batch is parent phi batch
};

class HierarchySparsingTheta : public RegularizerInterface {
 public:
  explicit HierarchySparsingTheta(const HierarchySparsingThetaConfig& config);

  virtual std::shared_ptr<RegularizeThetaAgent>
  CreateRegularizeThetaAgent(const Batch& batch, const ProcessBatchesArgs& args, float tau);

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  HierarchySparsingThetaConfig config_;
};

}  // namespace regularizer
}  // namespace artm
