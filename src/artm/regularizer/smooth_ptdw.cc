// Copyright 2017, Additive Regularization of Topic Models.

// Author: Anya Potapenko (anya_potapenko@mail.ru)

#include <vector>

#include "glog/logging.h"
#include "artm/core/protobuf_helpers.h"
#include "artm/utility/blas.h"

#include "artm/regularizer/smooth_ptdw.h"

namespace artm {
namespace regularizer {

void SmoothPtdwAgent::Apply(int item_index, int inner_iter, ::artm::utility::LocalPhiMatrix<float>* ptdw) const {
  int local_token_size = ptdw->num_tokens();
  int topic_size = ptdw->num_topics();

  if (config_.type() == SmoothPtdwConfig_SmoothType_MovingAverage) {
    // 1. evaluate wich tokens are background
    float threshold = config_.threshold();
    std::vector<bool> is_background(local_token_size, false);
    int count_background = 0;
    for (int i = 0; i < local_token_size; ++i) {
      const float* local_ptdw_ptr = &(*ptdw)(i, 0);  // NOLINT
      float sum_background = 0.0;
      for (int k = 0; k < topic_size; ++k) {
        char b = 'b';
        if (args_.topic_name(k)[0] == b) {  // background topic
          sum_background += local_ptdw_ptr[k];
        }
      }
      if (sum_background > threshold) {
        is_background[i] = true;
        ++count_background;
      }
    }

    // 2. prepare ptdw copy and smoothing profile
    int h = config_.window() / 2;
    ::artm::utility::LocalPhiMatrix<float> copy_ptdw(*ptdw);
    ::artm::utility::LocalPhiMatrix<float> smoothed(1, topic_size);
    smoothed.InitializeZeros();
    float* smoothed_ptr = &smoothed(0, 0);

    for (int i = 0; i < h && i < local_token_size; ++i) {
      if (is_background[i]) {
        continue;
      }

      const float* copy_ptdw_ptr = &copy_ptdw(i, 0);
      for (int k = 0; k < topic_size; ++k) {
        smoothed_ptr[k] += copy_ptdw_ptr[k];
      }
    }

    // 3. regularize
    for (int i = 0; i < local_token_size; ++i) {
      if (is_background[i]) {
        continue;
      }

      float* local_ptdw_ptr = &(*ptdw)(i, 0);  // NOLINT
      for (int k = 0; k < topic_size; ++k) {
        local_ptdw_ptr[k] += tau_ * smoothed_ptr[k];
        if (i + h < local_token_size && !is_background[i + h]) {
          smoothed_ptr[k] += copy_ptdw(i + h, k);
        }
        if (i - h >= 0 && !is_background[i - h]) {
          smoothed_ptr[k] -= copy_ptdw(i - h, k);
        }
      }
    }
  }

  // Multiplying neighbours (mode = 2)
  if (config_.type() == SmoothPtdwConfig_SmoothType_MovingProduct) {
    ::artm::utility::LocalPhiMatrix<float> copy_ptdw(*ptdw);
    for (int i = 0; i < local_token_size; ++i) {
      float* local_ptdw_ptr = &(*ptdw)(i, 0);  // NOLINT
      for (int k = 0; k < topic_size; ++k) {
        if (i + 1 < local_token_size) {
          local_ptdw_ptr[k] *= copy_ptdw(i + 1, k);
        }
        if (i - 1 >= 0) {
          local_ptdw_ptr[k] *= copy_ptdw(i - 1, k);
        }
      }
    }
  }
}

std::shared_ptr<RegularizePtdwAgent>
SmoothPtdw::CreateRegularizePtdwAgent(const Batch& batch,
                                      const ProcessBatchesArgs& args, float tau) {
  SmoothPtdwAgent* agent = new SmoothPtdwAgent(config_, args, tau);
  std::shared_ptr<RegularizePtdwAgent> retval(agent);
  return retval;
}

bool SmoothPtdw::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  SmoothPtdwConfig regularizer_config;
  if (!regularizer_config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse SmoothPtdwConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  return true;
}

}  // namespace regularizer
}  // namespace artm
