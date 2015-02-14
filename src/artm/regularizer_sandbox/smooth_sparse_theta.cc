// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include "artm/regularizer_sandbox/smooth_sparse_theta.h"

#include <vector>

#include "glog/logging.h"
#include "artm/core/protobuf_helpers.h"

namespace artm {
namespace regularizer_sandbox {

bool SmoothSparseTheta::RegularizeTheta(const Batch& batch,
                                        const ModelConfig& model_config,
                                        int inner_iter,
                                        double tau,
                                        artm::utility::DenseMatrix<float>* theta) {
  const int topic_size = theta->no_rows();
  const int item_size = theta->no_columns();

  if (item_size != batch.item_size()) {
    LOG(ERROR) << "theta->no_columns() != batch.item_size()";
    return false;
  }

  if (topic_size != model_config.topics_count()) {
    LOG(ERROR) << "theta->no_rows() != model_config.topics_count()";
    return false;
  }

  float alpha = 1.0f * tau;
  // *_size() starts from 1, inner_iter --- from 0
  if (config_.alpha_iter_size() >= (inner_iter + 1))
    alpha = tau * config_.alpha_iter(inner_iter);

  if (config_.topic_name_size() == 0) {
    // proceed the regularization
    float* data = theta->get_data();
    for (int i = 0; i < theta->size(); ++i)
      data[i] += alpha;
    return true;
  }

  if (topic_size != model_config.topic_name_size()) {
    LOG(ERROR) << "model_config.topics_count() != model_config.topic_name_size()";
    return false;
  }
  std::vector<int> topics_to_regularize(topic_size, false);
  for (int topic_id = 0; topic_id < config_.topic_name_size(); ++topic_id) {
    int topic_index = ::artm::core::repeated_field_index_of(
      model_config.topic_name(), config_.topic_name(topic_id));
    if (topic_index != -1) topics_to_regularize[topic_index] = true;
  }

  // proceed the regularization
  for (int item_index = 0; item_index < item_size; ++item_index) {
    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      if (topics_to_regularize[topic_id]) {
        (*theta)(topic_id, item_index) += alpha;
      }
    }
  }

  return true;
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
