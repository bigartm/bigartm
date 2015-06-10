// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <string>
#include <vector>

#include "artm/core/protobuf_helpers.h"
#include "artm/core/regularizable.h"
#include "artm/core/topic_model.h"

#include "artm/regularizer/improve_coherence_phi.h"

namespace artm {
namespace regularizer {

bool ImproveCoherencePhi::RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                                        const ::artm::core::PhiMatrix& n_wt,
                                        ::artm::core::PhiMatrix* result) {
  const int topic_size = n_wt.topic_size();
  const int token_size = n_wt.token_size();

  std::vector<bool> topics_to_regularize;
  if (config_.topic_name().size() == 0)
    topics_to_regularize.assign(topic_size, true);
  else
    topics_to_regularize = core::is_member(n_wt.topic_name(), config_.topic_name());

  bool use_all_classes = false;
  if (config_.class_id_size() == 0) {
    use_all_classes = true;
  }

  if (!config_.has_dictionary_name()) {
    LOG(WARNING) << "There's no dictionary for ImproveCoherence regularizer. Cancel it's launch.";
    return false;
  }

  auto dictionary_ptr = dictionary(config_.dictionary_name());
  if (dictionary_ptr == nullptr) {
    LOG(WARNING) << "There's no dictionary for ImproveCoherence regularizer. Cancel it's launch.";
    return false;
  }

  // proceed the regularization
  for (int token_id = 0; token_id < token_size; ++token_id) {
    auto token = n_wt.token(token_id);
    if (!use_all_classes && !core::is_member(token.class_id, config_.class_id())) continue;

    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      float value = 0.0f;
      auto& cooc_tokens_info = dictionary_ptr->cooc_info(token);
      for (int cooc_token_id = 0; cooc_token_id < dictionary_ptr->cooc_size(token); ++cooc_token_id) {
        if (cooc_tokens_info[cooc_token_id].token->class_id != token.class_id) continue;

        value += n_wt.get(n_wt.token_index(*cooc_tokens_info[cooc_token_id].token), topic_id) *
                 dictionary_ptr->cooc_value(token, cooc_token_id);
      }
      result->set(token_id, topic_id, value);
    }
  }
  return true;
}

google::protobuf::RepeatedPtrField<std::string> ImproveCoherencePhi::topics_to_regularize() {
  return config_.topic_name();
}

google::protobuf::RepeatedPtrField<std::string> ImproveCoherencePhi::class_ids_to_regularize() {
  return config_.class_id();
}

bool ImproveCoherencePhi::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  ImproveCoherencePhiConfig regularizer_config;
  if (!regularizer_config.ParseFromArray(config_blob.c_str(), config_blob.length())) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse ImproveCoherencePhiConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  return true;
}

}  // namespace regularizer
}  // namespace artm
