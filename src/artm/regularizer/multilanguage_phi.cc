// Copyright 2017, Additive Regularization of Topic Models.

#include <string>

#include "artm/core/phi_matrix.h"

#include "artm/regularizer/multilanguage_phi.h"

namespace artm {
namespace regularizer {

bool MultiLanguagePhi::RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                                     const ::artm::core::PhiMatrix& n_wt,
                                     ::artm::core::PhiMatrix* result) {
  // the body of this method will be defined later

  ++no_regularization_calls_;
  return true;
}

bool MultiLanguagePhi::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  MultiLanguagePhiConfig regularizer_config;
  if (!regularizer_config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse MultiLanguagePhiConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  return true;
}

}  // namespace regularizer
}  // namespace artm
