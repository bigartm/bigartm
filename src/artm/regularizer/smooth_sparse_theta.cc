// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <vector>

#include "glog/logging.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/regularizer/smooth_sparse_theta.h"

namespace artm {
namespace regularizer {

void SmoothSparseThetaAgent::Apply(int item_index, int inner_iter, int topics_size, float* theta) const {
  assert(topics_size == topic_weight.size());
  assert(inner_iter < alpha_weight.size());
  if (topics_size != topic_weight.size()) return;
  if (inner_iter >= alpha_weight.size()) return;

  for (int topic_id = 0; topic_id < topics_size; ++topic_id) {
    double value = transform_function_->apply(theta[topic_id]);
    theta[topic_id] += value > 0.0f ? alpha_weight[inner_iter] * topic_weight[topic_id] * value : 0.0f;
  }
}

SmoothSparseTheta::SmoothSparseTheta(const SmoothSparseThetaConfig& config)
    : config_(config),
      transform_function_(nullptr) {
  if (config.has_transform_config())
    transform_function_ = artm::core::TransformFunction::create(config.transform_config());
  else
    transform_function_ = artm::core::TransformFunction::create();
}

std::shared_ptr<RegularizeThetaAgent>
SmoothSparseTheta::CreateRegularizeThetaAgent(const Batch& batch,
                                              const ProcessBatchesArgs& args, double tau) {
  SmoothSparseThetaAgent* agent = new SmoothSparseThetaAgent(transform_function_);
  std::shared_ptr<SmoothSparseThetaAgent> retval(agent);

  const int topic_size = args.topic_name_size();
  const int item_size = batch.item_size();

  if (config_.alpha_iter_size() != 0) {
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

  agent->topic_weight.resize(topic_size, 0.0f);
  if (config_.topic_name_size() == 0) {
    for (int i = 0; i < topic_size; ++i)
      agent->topic_weight[i] = static_cast<float>(tau);
  } else {
    for (int topic_id = 0; topic_id < config_.topic_name_size(); ++topic_id) {
      int topic_index = ::artm::core::repeated_field_index_of(
        args.topic_name(), config_.topic_name(topic_id));
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
  if (!regularizer_config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse SmoothSparseThetaConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);

  if (config_.has_transform_config())
    transform_function_ = artm::core::TransformFunction::create(config_.transform_config());
  else
    transform_function_ = artm::core::TransformFunction::create();

  return true;
}

}  // namespace regularizer
}  // namespace artm
