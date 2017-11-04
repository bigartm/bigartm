// Copyright 2017, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <string>
#include <vector>

#include "artm/core/protobuf_helpers.h"
#include "artm/core/phi_matrix.h"

#include "artm/regularizer/improve_coherence_phi.h"

namespace artm {
namespace regularizer {

bool ImproveCoherencePhi::RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                                        const ::artm::core::PhiMatrix& n_wt,
                                        ::artm::core::PhiMatrix* result) {
  const int topic_size = n_wt.topic_size();
  const int token_size = n_wt.token_size();

  std::vector<bool> topics_to_regularize;
  if (config_.topic_name().size() == 0) {
    topics_to_regularize.assign(topic_size, true);
  } else {
    topics_to_regularize = core::is_member(n_wt.topic_name(), config_.topic_name());
  }

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

  // create the conversion from index if token in Dictionary -> index of token in Phi
  // -1 means that token from dictionary doesn't present in Phi matrix
  std::vector<int> dict_to_phi_indices(dictionary_ptr->size(), -1);
  for (int index = 0; index < dictionary_ptr->size(); ++index) {
    const auto& entry = dictionary_ptr->entries()[index];
    dict_to_phi_indices[index] = n_wt.token_index(entry.token());
  }

  // proceed the regularization
  for (int token_id = 0; token_id < token_size; ++token_id) {
    const auto& token = n_wt.token(token_id);
    if (!use_all_classes && !core::is_member(token.class_id, config_.class_id())) {
      continue;
    }

    auto cooc_tokens_info = dictionary_ptr->token_cooc_values(token);
    if (cooc_tokens_info == nullptr) {
      continue;
    }

    std::vector<float> values(topic_size, 0.0);
    for (const auto& elem : *cooc_tokens_info) {
      float mult_coef = elem.second;
      int cooc_token_index = dict_to_phi_indices[elem.first];
      if (cooc_token_index == -1) {
        continue;
      }

      for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
        if (!topics_to_regularize[topic_id]) {
          continue;
        }

        values[topic_id] += n_wt.get(cooc_token_index, topic_id) * mult_coef;
      }
    }
    result->increase(token_id, values);
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
  if (!regularizer_config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse ImproveCoherencePhiConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  return true;
}

}  // namespace regularizer
}  // namespace artm
