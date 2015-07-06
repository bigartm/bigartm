// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <cmath>
#include <memory>
#include <vector>

#include "artm/core/protobuf_helpers.h"
#include "artm/core/topic_model.h"

#include "artm/score/coherence_plugin.h"

namespace artm {
namespace score {

float CountTopicCoherence(const std::shared_ptr<core::Dictionary>& dictionary,
                          const std::vector<core::Token>& tokens_to_score) {
  float coherence_value = 0.0;
  int k = tokens_to_score.size();
  if (k == 0) return 0.0f;

  for (int i = 0; i < k - 1; ++i) {
    for (int j = i; j < k; ++j) {
      if (tokens_to_score[j].class_id != tokens_to_score[i].class_id) continue;
      coherence_value += static_cast<float>(dictionary->cooc_value(tokens_to_score[i], tokens_to_score[j]));
    }
  }

  return 2 / (k * (k - 1)) * coherence_value;
}

}  // namespace score
}  // namespace artm
