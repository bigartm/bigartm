// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include "artm/regularizer_sandbox/smooth_sparse_theta.h"

#include <vector>

namespace artm {
namespace regularizer_sandbox {

bool SmoothSparseTheta::RegularizeTheta(const Item& item,
                                        std::vector<float>* n_dt,
                                        google::protobuf::RepeatedPtrField<std::string> topic_name,
                                        int inner_iter,
                                        double tau) {
  // read the parameters from config and control their correctness
  const int topic_size = topic_name.size();
  std::vector<bool> topics_to_regularize;

  if (config_.topic_name_size() > 0) {
    for (int i = 0; i < topic_size; ++i)
      topics_to_regularize.push_back(false);

    for (int topic_id = 0; topic_id < config_.topic_name_size(); ++topic_id) {
      for (int real_topic_id = 0; real_topic_id < topic_size; ++real_topic_id) {
        if (topic_name.Get(real_topic_id) == config_.topic_name(topic_id)) {
          topics_to_regularize[real_topic_id] = true;
          break;
        }
      }
    }
  } else {
    for (int i = 0; i < topic_size; ++i)
      topics_to_regularize.push_back(true);
  }

  float cur_iter_alpha = 1;
  // *_size() starts from 1, inner_iter --- from 0
  if (config_.alpha_iter_size() >= inner_iter + 1)
    cur_iter_alpha = config_.alpha_iter(inner_iter);

  // proceed the regularization
  for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
    if (topics_to_regularize[topic_id])
      (*n_dt)[topic_id] = (*n_dt)[topic_id] + static_cast<float>(tau * cur_iter_alpha);
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
