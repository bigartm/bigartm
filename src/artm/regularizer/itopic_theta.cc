// Copyright 2014, Additive Regularization of Topic Models.

// Author: me

#include <vector>

#include "glog/logging.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/regularizer/itopic_theta.h"

iTopicThetaAgent
iTopicTheta

namespace artm {
namespace regularizer {

void iTopicThetaAgent::Apply(int item_index, int inner_iter, int topics_size, float* theta) const {
  auto phi = GetPhiMatrix(instance_->config()->nwt_name());
  // now we need to get tokens inside current document
  const Item& item = mybatch.item(item_index);
  for (int token_index = 0; token_index < item.token_id_size(); ++token_index) {
    int token_id = item.token_id(token_index);
    ClassId class_id = batch.class_id(token_id);
    if (class_id == config_.class_name) {
      float token_weight = item.token_weight(token_index);
      // auto iter = class_id_to_weight.find(class_id);
      // float class_weight = (iter == class_id_to_weight.end()) ? 0.0f : iter->second;

      // NOTE: we could use class_weight * token_weight here instead
      // but that makes combining iTopicTheta with log-likelihood 
      // regularizer more tricky for no actual gain
      for (int topic_id = 0; topic_id < topics_size; ++topic_id) {
        theta[topic_id] += token_weight * phi->get(token_index, topic_index);
      }
    }
  }
}

iTopicTheta::iTopicTheta(const iTopicThetaConfig& config) : config_(config) { }

std::shared_ptr<RegularizeThetaAgent>
TopicSelectionTheta::CreateRegularizeThetaAgent(const Batch& batch,
                                                const ProcessBatchesArgs& args, double tau) {
  TopicSelectionThetaAgent* agent = new TopicSelectionThetaAgent();
  std::shared_ptr<TopicSelectionThetaAgent> retval(agent);

  const int topic_size = args.topic_name_size();
  const int item_size = batch.item_size();

  if (config_.alpha_iter_size()) {
    if (args.num_document_passes() != config_.alpha_iter_size()) {
      LOG(ERROR) << "ProcessBatchesArgs.num_document_passes() != SmoothSparseThetaConfig.alpha_iter_size()";
      return nullptr;
    }

    for (int i = 0; i < config_.alpha_iter_size(); ++i)
      agent->alpha_weight.push_back(config_.alpha_iter(i));
  } else {
    for (int i = 0; i < args.num_document_passes(); ++i)
      agent->alpha_weight.push_back(1.0f);
  }

  if (config_.topic_value_size()) {
    if (topic_size != config_.topic_value_size()) {
      LOG(ERROR) << "ProcessBatchesArgs.num_topics() != TopicSelectionThetaConfig.topic_value_size()";
      return nullptr;
    }

    for (int i = 0; i < topic_size; ++i)
      agent->topic_value.push_back(config_.topic_value(i));
  } else {
    for (int i = 0; i < topic_size; ++i)
      agent->topic_value.push_back(1.0f);
  }

  agent->topic_weight.resize(topic_size, 0.0f);
  if (config_.topic_name_size() == 0) {
    for (int i = 0; i < topic_size; ++i)
      agent->topic_weight[i] = static_cast<float>(-tau);
  } else {
    if (topic_size != args.topic_name_size()) {
      LOG(ERROR) << "args.num_topics() != args.topic_name_size()";
      return nullptr;
    }

    for (int topic_id = 0; topic_id < config_.topic_name_size(); ++topic_id) {
      int topic_index = ::artm::core::repeated_field_index_of(
        args.topic_name(), config_.topic_name(topic_id));
      if (topic_index != -1) agent->topic_weight[topic_index] = static_cast<float>(-tau);
    }
  }

  return retval;
}

google::protobuf::RepeatedPtrField<std::string> TopicSelectionTheta::topics_to_regularize() {
  return config_.topic_name();
}

bool TopicSelectionTheta::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  TopicSelectionThetaConfig regularizer_config;
  if (!regularizer_config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse TopicSelectionThetaConfig from RegularizerConfig.config"));
  }
  config_.CopyFrom(regularizer_config);

  return true;
}

}  // namespace regularizer
}  // namespace artm
