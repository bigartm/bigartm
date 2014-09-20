// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include "artm/regularizer_sandbox/smooth_sparse_theta.h"

#include <string>
#include <vector>

namespace artm {
namespace regularizer_sandbox {

bool SmoothSparseTheta::RegularizeTheta(const Item& item,
                                     std::vector<float>* n_dt,
                                     int topic_size,
                                     int inner_iter,
                                     double tau) {
  // read the parameters from config and control their correctness
  int background_topics_count = 0;
  if (config_.has_background_topics_count()) {
    background_topics_count = config_.background_topics_count();
  }

  if (background_topics_count < 0 || background_topics_count > topic_size) {
    LOG(ERROR) << "Smooth/Sparse Theta: background_topics_count must be in [0, topics_size]";
    return false;
  }
  const int objective_topic_size = topic_size - background_topics_count;

  ::artm::FloatArray alpha_topic;
  if (config_.has_alpha_topic()) {
    alpha_topic.CopyFrom(config_.alpha_topic());

    if (alpha_topic.value_size() != topic_size) {
      LOG(ERROR) << "Smooth/Sparse Theta: len(alpha_topic) must be equal to len(topic_size)";
      return false;
    }
  } else {
    // make default values
    for (int i = 0; i < objective_topic_size; ++i) {
      alpha_topic.add_value(-1);
    }

    for (int i = objective_topic_size; i < topic_size; ++i) {
      alpha_topic.add_value(+1);
    }
  }

  float cur_iter_coef = 1;
  if (config_.has_alpha_iter()) {
    auto alpha_iter = config_.alpha_iter();
    // value_size() start from 1, inner_iter --- from 0
    if (alpha_iter.value_size() >= inner_iter + 1) {
      cur_iter_coef = alpha_iter.value().Get(inner_iter);
    }
  }

  // proceed the regularization
  for (int i = 0; i < topic_size; ++i) {
    (*n_dt)[i] = (*n_dt)[i] + static_cast<float>(tau * cur_iter_coef * alpha_topic.value().Get(i));
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
