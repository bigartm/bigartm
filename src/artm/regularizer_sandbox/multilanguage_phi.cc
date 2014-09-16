// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/regularizer_sandbox/multilanguage_phi.h"

#include <string>

#include "artm/core/regularizable.h"
#include "artm/core/topic_model.h"

namespace artm {
namespace regularizer_sandbox {

bool MultiLanguagePhi::RegularizePhi(::artm::core::Regularizable* topic_model, double tau) {
  // the body of this method will be defined later

  no_regularization_calls_++;
  return true;
}

bool MultiLanguagePhi::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  MultiLanguagePhiConfig regularizer_config;
  if (!regularizer_config.ParseFromArray(config_blob.c_str(), config_blob.length())) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse MultiLanguagePhiConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  return true;
}

void MultiLanguagePhi::SerializeInternalState(RegularizerInternalState* regularizer_state) {
  MultiLanguagePhiInternalState data;
  data.set_no_regularization_calls(no_regularization_calls_);
  regularizer_state->set_type(RegularizerInternalState_Type_MultiLanguagePhi);
  regularizer_state->set_data(data.SerializeAsString());
}

}  // namespace regularizer_sandbox
}  // namespace artm
