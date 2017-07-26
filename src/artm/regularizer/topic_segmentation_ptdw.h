/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Anastasia Bayandina (anast.bayandina@gmail.com)

   ToDo: Description will be updated later
*/

#pragma once

#include <string>
#include <vector>

#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class TopicSegmentationPtdwAgent : public RegularizePtdwAgent {
 private:
  friend class TopicSegmentationPtdw;
  TopicSegmentationPtdwConfig config_;
  ProcessBatchesArgs args_;

 public:
  TopicSegmentationPtdwAgent(const TopicSegmentationPtdwConfig& config, const ProcessBatchesArgs& args, float tau)
      : config_(config)
      , args_(args) { }

  virtual void Apply(int item_index, int inner_iter, ::artm::utility::LocalPhiMatrix<float>* ptdw) const;
};

class TopicSegmentationPtdw : public RegularizerInterface {
 public:
  explicit TopicSegmentationPtdw(const TopicSegmentationPtdwConfig& config) : config_(config) { }

  virtual std::shared_ptr<RegularizePtdwAgent>
  CreateRegularizePtdwAgent(const Batch& batch, const ProcessBatchesArgs& args, float tau);

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  TopicSegmentationPtdwConfig config_;
};

}  // namespace regularizer
}  // namespace artm
