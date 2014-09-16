// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include "artm/regularizer_sandbox/dirichlet_theta.h"

#include <string>
#include <vector>

namespace artm {
namespace regularizer_sandbox {

bool DirichletTheta::RegularizeTheta(const Item& item,
                                     std::vector<float>* n_dt,
                                     int topic_size,
                                     int inner_iter,
                                     double tau) {
  const ::google::protobuf::RepeatedPtrField<DoubleArray>& alpha_vector =
    config_.alpha();

  // inner_iter is from [0 iter_num]
  if (alpha_vector.size() < inner_iter + 1) {
    for (int i = 0; i < topic_size; ++i) {
      (*n_dt)[i] = (*n_dt)[i] + static_cast<float>(tau * 1);
    }
  } else {
    const artm::DoubleArray& alpha = alpha_vector.Get(inner_iter);
    if (alpha.value_size() == topic_size) {
      for (int i = 0; i < topic_size; ++i) {
        (*n_dt)[i] = (*n_dt)[i] + static_cast<float>(tau * alpha.value().Get(i));
      }
    } else {
      return false;
    }
  }
  return true;
}

bool DirichletTheta::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  DirichletThetaConfig regularizer_config;
  if (!regularizer_config.ParseFromArray(config_blob.c_str(), config_blob.length())) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse DecorrelatorThetaConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  return true;
}

}  // namespace regularizer_sandbox
}  // namespace artm
