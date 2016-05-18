// Copyright 2014, Additive Regularization of Topic Models.

// Author: me

#include <vector>

#include "glog/logging.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/regularizer/itopic_theta.h"

namespace artm {
namespace regularizer {


void iTopicThetaAgent::Apply(int item_index, int inner_iter, int topics_size, float* theta) const {
  auto phi = myinstance_->GetPhiMatrixSafe(myinstance_->config()->nwt_name());
  // now we need to get tokens inside current document
  const Item& item = mybatch.item(item_index);
  for (int token_index = 0; token_index < item.token_id_size(); ++token_index) {
    int token_id = item.token_id(token_index);
    ::artm::core::ClassId class_id = mybatch.class_id(token_id);
    std::cout << "class_id of [" << token_id << "] is " << class_id << std::endl;
    if (class_id == myclass) {
      std::cout << "OK" << std::endl;

      float token_weight = item.token_weight(token_index);

      // NOTE: we could use class_weight * token_weight here instead
      // but that makes combining iTopicTheta with log-likelihood 
      // regularizer more tricky for no actual gain
      for (int topic_id = 0; topic_id < topics_size; ++topic_id) {
        theta[topic_id] += token_weight * phi->get(token_index, topic_id);
      }
    }
  }
}

iTopicTheta::iTopicTheta(const iTopicThetaConfig& config) : config_(config) { }

std::shared_ptr<RegularizeThetaAgent>
iTopicTheta::CreateRegularizeThetaAgent(const Batch& batch,
                                                const ProcessBatchesArgs& args, double tau) {
  std::shared_ptr<const ::artm::core::PhiMatrix> GetPhiMatrix(const std::string& model_name);
  iTopicThetaAgent* agent = new iTopicThetaAgent(batch, instance_);
  std::shared_ptr<iTopicThetaAgent> retval(agent);

  const int topic_size = args.topic_name_size();
  const int item_size = batch.item_size();
  // agent->myclass = config_.class_name();

  agent->myclass = config_.class_name();
  std::cout << "myclass is " << config_.class_name() << std::endl;

  
  // TODO: various checks here
  // use_classes == true, class_name is valid, class_weight != 0, etc
  // auto iter = class_id_to_weight.find(config_.class_name);
  // float class_weight = (iter == class_id_to_weight.end()) ? 0.0f : iter->second;
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
