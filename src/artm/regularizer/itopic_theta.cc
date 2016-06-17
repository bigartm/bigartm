// Copyright 2014, Additive Regularization of Topic Models.

// Author: Bulatov Victor

#include <vector>

#include "glog/logging.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/regularizer/itopic_theta.h"

namespace artm {
namespace regularizer {


void iTopicThetaAgent::Apply(int item_index, int inner_iter, int topics_size, float* theta) const {

  auto phi = myinstance_->GetPhiMatrixSafe(config_.nwt_name());
  // now we need to get tokens inside current document
  const Item& item = mybatch.item(item_index);
  for (int token_index = 0; token_index < item.token_id_size(); ++token_index) {
    int token_id = item.token_id(token_index);
    ::artm::core::ClassId class_id = mybatch.class_id(token_id);
    if (class_id == config_.class_name()) {
      float token_weight = item.token_weight(token_index);
      // NOTE: why I use token_weight instead of class_weight * token_weight?
      // Short version: adjusting for class_weight is easier than un-adjusting. 
      // And damages perfomance (in a certain sense).

      for (int topic_id = 0; topic_id < topics_size; ++topic_id) {
        theta[topic_id] += tau_ * token_weight * phi->get(token_id, topic_id);
      }
    }
  }
}


std::shared_ptr<RegularizeThetaAgent>
iTopicTheta::CreateRegularizeThetaAgent(const Batch& batch,
                                                const ProcessBatchesArgs& args, double tau) {
  iTopicThetaAgent* agent = new iTopicThetaAgent(batch, instance_);
  std::shared_ptr<iTopicThetaAgent> retval(agent);

  const int topic_size = args.topic_name_size();
  const int item_size = batch.item_size();
  agent->config_ = config_;
  agent->tau_ = tau;

  
  // TODO: various checks here
  // use_classes == true, class_name is valid, class_weight != 0, etc
  return retval;
}


bool iTopicTheta::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  iTopicThetaConfig regularizer_config;
  if (!regularizer_config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse iTopicThetaConfig from RegularizerConfig.config"));
  }
  config_.CopyFrom(regularizer_config);
  
  return true;
}
}  // namespace regularizer
}  // namespace artm
