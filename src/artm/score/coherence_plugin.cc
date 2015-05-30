// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <cmath>
#include <memory>
#include <vector>

#include "artm/core/exceptions.h"
#include "artm/core/protobuf_helpers.h"
#include "artm/core/topic_model.h"

#include "artm/score/coherence_plugin.h"

namespace artm {
namespace score {

float CountTopicCoherence(const std::shared_ptr<core::Dictionary>& dictionary,
                          const std::vector<core::Token>& tokens_to_score) {
  float coherence_value = 0.0;
  if (tokens_to_score.size() == 0) return 0.0f;

  for (int i = 0; i < tokens_to_score.size() - 1; ++i) {
    auto entry_ptr_one = dictionary->entry(tokens_to_score[i]);
    if (entry_ptr_one == nullptr) {
      LOG(WARNING) << "CountTopicCoherence: token (" << tokens_to_score[i].keyword <<
        ", " << tokens_to_score[i].class_id << ") doesn't present in dictionary and will be skiped";
      continue;
    }

    for (int j = i; j < tokens_to_score.size(); ++j) {
      auto entry_ptr_two = dictionary->entry(tokens_to_score[j]);
      if (entry_ptr_two == nullptr) {
        LOG(WARNING) << "CountTopicCoherence: token (" << tokens_to_score[j].keyword <<
          ", " << tokens_to_score[j].class_id << ") doesn't present in dictionary and will be skiped";
        continue;
      }

      int denominator = 1;
      if (entry_ptr_one->has_items_count()) {
        denominator *= entry_ptr_one->items_count();
      } else {
        LOG(WARNING) << "CountTopicCoherence: token (" << tokens_to_score[i].keyword <<
          ", " << tokens_to_score[i].class_id << ") doesn't heve items_count info in dictionary and will be skiped";
        continue;
      }

      if (entry_ptr_two->has_items_count()) {
        denominator *= entry_ptr_two->items_count();
      } else {
        LOG(WARNING) << "CountTopicCoherence: token (" << tokens_to_score[j].keyword <<
          ", " << tokens_to_score[j].class_id << ") doesn't heve items_count info in dictionary and will be skiped";
        continue;
      }

      float numerator = static_cast<float>(dictionary->cooc_value(tokens_to_score[i], tokens_to_score[j]));
      coherence_value += denominator == 0 || std::fabs(numerator) < 1e-37 ? 0.0 :
        std::log(numerator / denominator * dictionary->total_items_count());
    }
  }

  return coherence_value;
}

}  // namespace score
}  // namespace artm
