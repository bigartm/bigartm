// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <cmath>

#include "artm/core/exceptions.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/score/background_tokens_ratio.h"

namespace artm {
namespace score {

std::shared_ptr<Score> BackgroundTokensRatio::CalculateScore(const artm::core::PhiMatrix& p_wt) {
  int topic_size = p_wt.topic_size();
  int token_size = p_wt.token_size();

  // parameters preparation
  double delta_threshold = config_.delta_threshold();
  if (delta_threshold < 0) {
    BOOST_THROW_EXCEPTION(artm::core::ArgumentOutOfRangeException(
      "BackgroundTokensRatioScoreConfig.delta_threshold",
      config_.delta_threshold()));
  }

  bool direct_kl = config_.direct_kl();
  bool save_tokens = config_.save_tokens();

  ::artm::core::ClassId class_id = ::artm::core::DefaultClass;
  if (config_.has_class_id())
    class_id = config_.class_id();

  // count score
  auto btp_score = new BackgroundTokensRatioScore();
  std::shared_ptr<Score> retval(btp_score);

  auto n_wt = GetPhiMatrix(instance_->config()->nwt_name());
  std::vector<double> n_t(topic_size, 0.0);
  std::vector<std::vector<core::Token> > topic_kernel_tokens(
    topic_size, std::vector<core::Token>());

  double n = 0.0;
  for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
    for (int token_index = 0; token_index < token_size; ++token_index) {
      double value = static_cast<double>(n_wt->get(token_index, topic_index));
      n_t[topic_index] += value;
      n += value;
    }
  }

  int num_bgr_tokens = 0;
  std::vector<artm::core::Token> bcg_tokens;
  for (int token_index = 0; token_index < token_size; ++token_index) {
    auto token = p_wt.token(token_index);
    if (token.class_id == class_id) {
      double p_w = 0.0;
      for (int topic_index = 0; topic_index < topic_size; ++topic_index)
        p_w += static_cast<double>(p_wt.get(token_index, topic_index) * n_t[topic_index]);
      p_w = (n > 0.0) ? p_w / n : 0.0;

      double kl_value = 0.0;
      if (p_w > 0) {
        for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
          double value = static_cast<double>(p_wt.get(token_index, topic_index));
          double p_t = n_t[topic_index] / n;
          double p_tw = value * p_t / p_w;

          if (p_tw > 0.0)
            kl_value += direct_kl ? (p_t * log(p_t / p_tw)) : (p_tw * log(p_tw / p_t));
        }

        if (kl_value > delta_threshold) {
          num_bgr_tokens++;
          if (save_tokens)
            bcg_tokens.push_back(token);
        }
      }
    }
  }

  btp_score->set_value(token_size > 0 ? (static_cast<float>(num_bgr_tokens) / token_size) : 0.0f);
  for (auto token : bcg_tokens)
    btp_score->add_token(token.keyword);

  return retval;
}

}  // namespace score
}  // namespace artm
