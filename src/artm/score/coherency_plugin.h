// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#ifndef SRC_ARTM_SCORE_COHERENCY_PLUGIN_H_
#define SRC_ARTM_SCORE_COHERENCY_PLUGIN_H_

#include <memory>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/score_calculator_interface.h"

namespace artm {
namespace score {

float CountTopicCoherency(const std::shared_ptr<core::Dictionary>& dictionary,
                          const std::vector<core::Token>& tokens_to_score);

}  // namespace score
}  // namespace artm

#endif  // SRC_ARTM_SCORE_COHERENCY_PLUGIN_H_
