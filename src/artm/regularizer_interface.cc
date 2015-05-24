// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/regularizer_interface.h"

#include "artm/core/dictionary.h"
#include "artm/core/topic_model.h"

namespace artm {

std::shared_ptr< ::artm::core::Dictionary> RegularizerInterface::dictionary(
    const std::string& dictionary_name) {
  if (dictionaries_ == nullptr) {
      return nullptr;
  }

  return dictionaries_->get(dictionary_name);
}

void RegularizerInterface::set_dictionaries(
    const ::artm::core::ThreadSafeDictionaryCollection* dictionaries) {
  dictionaries_ = dictionaries;
}

void RegularizeThetaAgentCollection::Apply(int item_index, int inner_iter, int topics_size, float* theta) {
  for (auto& agent : agents_) {
    if (agent != nullptr) {
      agent->Apply(item_index, inner_iter, topics_size, theta);
    }
  }
}

void NormalizeThetaAgent::Apply(int item_index, int inner_iter, int topics_size, float* theta) {
  float sum = 0.0f;
  for (int topic_index = 0; topic_index < topics_size; ++topic_index) {
    float val = theta[topic_index];
    if (val > 0)
      sum += val;
  }

  float sum_inv = sum > 0.0f ? (1.0f / sum) : 0.0f;
  for (int topic_index = 0; topic_index < topics_size; ++topic_index) {
    float val = sum_inv * theta[topic_index];
    if (val < 1e-16f) val = 0.0f;
    theta[topic_index] = val;
  }
}


}  // namespace artm
