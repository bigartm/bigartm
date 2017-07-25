// Copyright 2017, Additive Regularization of Topic Models.

// Author: Anastasia Bayandina (anast.bayandina@gmail.com)

#include <vector>
#include <algorithm>

#include "glog/logging.h"
#include "artm/core/protobuf_helpers.h"
#include "artm/utility/blas.h"

#include "artm/regularizer/topic_segmentation_ptdw.h"

namespace artm {
namespace regularizer {

void TopicSegmentationPtdwAgent::Apply(int item_index, int inner_iter,
                                       ::artm::utility::LocalPhiMatrix<float>* ptdw) const {
  const int local_token_size = ptdw->no_rows();
  const int topic_size = ptdw->no_columns();
  std::vector<float> background_probability(local_token_size, 0.0f);

  // if background topics are given, count probability for each word to be background
  if (config_.background_topic_names().size()) {
    std::vector<bool> is_background_topic = core::is_member(args_.topic_name(), config_.background_topic_names());
    for (int i = 0; i < local_token_size; ++i) {
      const float* local_ptdw_ptr = &(*ptdw)(i, 0);  // NOLINT
      for (int k = 0; k < topic_size; ++k) {
        if (is_background_topic[k]) {
          background_probability[i] += local_ptdw_ptr[k];
        }
      }
    }
  }

  int h = config_.window();
  float threshold_topic_change = config_.threshold();
  ::artm::utility::DenseMatrix<float> copy_ptdw(*ptdw);
  std::vector<float> left_distribution(topic_size, 0.0f);
  std::vector<float> right_distribution(topic_size, 0.0f);

  float left_weights = 0.0f;
  float right_weights = 0.0f;
  int l_topic, r_topic;  // topic ids on which maximum of the distribution is reached
  bool changes_topic = false;
  int main_topic = 0;

  auto main_topic_density = (*ptdw)(0, 0);
  for (int k = 0; k < topic_size; ++k) {
    auto cur_topic_density = (*ptdw)(0, k);
    if (main_topic_density < cur_topic_density) {
      main_topic_density = cur_topic_density;
      main_topic = k;
    }
  }
  for (int k = 0; k < topic_size; ++k) {
    if (k == main_topic) {
      (*ptdw)(0, k) = 1.0f;
    } else {
      (*ptdw)(0, k) = 0.0f;
    }
  }
  for (int i = 0; i < h && i < local_token_size; ++i) {
    for (int k = 0; k < topic_size; ++k) {
      right_distribution[k] += copy_ptdw(i, k) * (1 - background_probability[i]);
    }
    right_weights += 1 - background_probability[i];
  }
  for (int i = 1; i < local_token_size; ++i) {
    for (int k = 0; k < topic_size; ++k) {
      left_distribution[k] += copy_ptdw(i - 1, k) * (1 - background_probability[i - 1]);
      right_distribution[k] -= copy_ptdw(i - 1, k) * (1 - background_probability[i - 1]);
    }
    left_weights += 1 - background_probability[i - 1];
    right_weights -= 1 - background_probability[i - 1];
    if (i <= local_token_size - h) {
      for (int k = 0; k < topic_size; ++k) {
        right_distribution[k] += copy_ptdw(i + h - 1, k) * (1 - background_probability[i + h - 1]);
      }
      right_weights += 1 - background_probability[i + h - 1];
    }
    if (i > h) {
      for (int k = 0; k < topic_size; ++k) {
        left_distribution[k] -= copy_ptdw(i - h - 1, k) * (1 - background_probability[i - h - 1]);
      }
      left_weights -= 1 - background_probability[i - h - 1];
    }
    auto lb = left_distribution.begin();
    auto le = left_distribution.end();
    auto rb = right_distribution.begin();
    auto re = right_distribution.end();

    l_topic = std::distance(lb, std::max_element(lb, le));
    r_topic = std::distance(rb, std::max_element(rb, re));

    float ll =  left_distribution[l_topic];
    float rl = right_distribution[l_topic];
    float rr = right_distribution[r_topic];
    float lr =  left_distribution[r_topic];

    changes_topic = ((ll / left_weights  - rl / right_weights) / 2 +
                     (rr / right_weights - lr / left_weights)  / 2 > threshold_topic_change);
    if (changes_topic) {
      main_topic = r_topic;
    }
    for (int k = 0; k < topic_size; ++k) {
      if (k == main_topic) {
        (*ptdw)(i, k) = 1.0f;
      } else {
        (*ptdw)(i, k) = 0.0f;
      }
    }
  }
}

std::shared_ptr<RegularizePtdwAgent>
TopicSegmentationPtdw::CreateRegularizePtdwAgent(const Batch& batch,
                                                 const ProcessBatchesArgs& args, float tau) {
  TopicSegmentationPtdwAgent* agent = new TopicSegmentationPtdwAgent(config_, args, tau);
  std::shared_ptr<RegularizePtdwAgent> retval(agent);
  return retval;
}

bool TopicSegmentationPtdw::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  TopicSegmentationPtdwConfig regularizer_config;
  if (!regularizer_config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse TopicSegmentationPtdwConfig from RegularizerConfig.config"));
  }
  config_.CopyFrom(regularizer_config);
  return true;
}

}  // namespace regularizer
}  // namespace artm
