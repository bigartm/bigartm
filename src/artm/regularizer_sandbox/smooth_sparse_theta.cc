// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include "artm/regularizer_sandbox/smooth_sparse_theta.h"

#include <vector>

#include "glog/logging.h"
#include "artm/core/protobuf_helpers.h"

namespace artm {
namespace regularizer_sandbox {

void SmoothSparseThetaAgent::Apply(int item_index, int inner_iter, int topics_size, float* theta) {
  assert(topics_size == topic_weight.size());
  assert(inner_iter < alpha_weight.size());
  if (topics_size != topic_weight.size()) return;
  if (inner_iter >= alpha_weight.size()) return;

  for (int topic_id = 0; topic_id < topics_size; ++topic_id)
    theta[topic_id] += alpha_weight[inner_iter] * topic_weight[topic_id];
}

std::shared_ptr<RegularizeThetaAgent>
SmoothSparseTheta::CreateRegularizeThetaAgent(const Batch& batch,
                                              const ModelConfig& model_config, double tau) {
  SmoothSparseThetaAgent* agent = new SmoothSparseThetaAgent();
  std::shared_ptr<SmoothSparseThetaAgent> retval(agent);

  const int topic_size = model_config.topics_count();
  const int item_size = batch.item_size();

  if (config_.alpha_iter_size() != 0) {
    if (model_config.inner_iterations_count() != config_.alpha_iter_size()) {
      LOG(ERROR) << "ModelConfig.inner_iterations_count() != SmoothSparseThetaConfig.alpha_iter_size()";
      return nullptr;
    }

    for (int i = 0; i < config_.alpha_iter_size(); ++i)
      agent->alpha_weight.push_back(config_.alpha_iter(i));
  } else {
    for (int i = 0; i < model_config.inner_iterations_count(); ++i)
      agent->alpha_weight.push_back(1.0f);
  }

  agent->topic_weight.resize(topic_size, 0.0f);
  if (config_.topic_name_size() == 0) {
    for (int i = 0; i < topic_size; ++i)
      agent->topic_weight[i] = static_cast<float>(tau);
  } else {
    if (topic_size != model_config.topic_name_size()) {
      LOG(ERROR) << "model_config.topics_count() != model_config.topic_name_size()";
      return nullptr;
    }

    for (int topic_id = 0; topic_id < config_.topic_name_size(); ++topic_id) {
      int topic_index = ::artm::core::repeated_field_index_of(
        model_config.topic_name(), config_.topic_name(topic_id));
      if (topic_index != -1) agent->topic_weight[topic_index] = static_cast<float>(tau);
    }
  }

  return retval;
}

google::protobuf::RepeatedPtrField<std::string> SmoothSparseTheta::topics_to_regularize() {
  return config_.topic_name();
}

bool SmoothSparseTheta::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  SmoothSparseThetaConfig regularizer_config;
  if (!regularizer_config.ParseFromArray(config_blob.c_str(), config_blob.length())) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse SmoothSparseThetaConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  return true;
}

}  // namespace regularizer_sandbox
}  // namespace artm
