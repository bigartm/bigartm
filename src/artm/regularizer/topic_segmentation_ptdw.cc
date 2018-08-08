// Copyright 2018, Additive Regularization of Topic Models.

// Author: Nikolay Skachkov

#include <vector>
#include <algorithm>
#include <cmath>

#include "glog/logging.h"
#include "artm/core/protobuf_helpers.h"
#include "artm/utility/blas.h"

#include "artm/regularizer/topic_segmentation_ptdw.h"

namespace artm {
namespace regularizer {


void TopicSegmentationPtdwAgent::Apply(int item_index, int inner_iter,
                                       ::artm::utility::LocalPhiMatrix<float>* ptdw) const {
  const int local_token_size = ptdw->no_rows();
  const int num_topics = ptdw->no_columns();
  std::vector<float> background_probability(local_token_size, 0.0f);
  //// if background topics are given, count probability for each word to be background
  std::vector<bool> is_background_topic = core::is_member(args_.topic_name(), config_.background_topic_names());
  if (config_.background_topic_names().size()) {
    for (int i = 0; i < local_token_size; ++i) {
      const float* local_ptdw_ptr = &(*ptdw)(i, 0);  // NOLINT
      for (int k = 0; k < num_topics; ++k) {
        if (is_background_topic[k]) {
          background_probability[i] += local_ptdw_ptr[k];
        }
      }
    }
  }

  ::artm::utility::DenseMatrix<float> copy_ptdw(*ptdw);
  float tau = tau_;
  float alpha = config_.merge_threshold();
  bool merge_into_segments = config_.merge_into_segments();
  std::list<int>& dot_positions = dot_positions_[item_index];
  float average_len = 0.0f;
  int seg_begin = 0;
  for (auto it = dot_positions.begin(); it != dot_positions.end(); it++) {
    average_len += *it - seg_begin;
    seg_begin = *it + 1;
  }
  average_len /= dot_positions.size();
  int sen_begin = 0;
  auto it = dot_positions.begin();
  std::vector<float> prev_sen_subject(num_topics, 0.0f);
  std::vector<float> distances(0, 0.0f);
  bool handling_first_sen = true;
  int prev_seg_len = 0;
  while (sen_begin < local_token_size) {
    int sen_end = *it;
    int sen_len = sen_end - sen_begin;
    if (sen_len <= 0) {
      sen_begin = sen_end + 1;
      it++;
      continue;
    }
    std::vector<float> sen_subj(num_topics, 0.0f);
    float norm_sum = 0.0f;
    for (int i = sen_begin; i < sen_end; i++) {
      float weight = 1.0f - background_probability[i];
      if (weight == 0.0f) {
        continue;
      }
      norm_sum += weight * weight;
      for (int t = 0; t < num_topics; t++) {
        if (is_background_topic[t]) {
          continue;
        }
        sen_subj[t] += copy_ptdw(i, t) * weight;
      }
    }
    for (int t = 0; t < num_topics; t++) {
      if (is_background_topic[t]) {
        continue;
      }
      if (norm_sum != 0) {
        sen_subj[t] /= norm_sum;
      }
    }
    for (int i = sen_begin; i < sen_end; i++) {
      float word_on_sent_contribution = 0.0f;
      float weight = 1.0f - background_probability[i];
      for (int t = 0; t < num_topics; t++) {
        if (is_background_topic[t]) {
          continue;
        }
        if (copy_ptdw(i, t) != 0 && sen_subj[t] != 0) {
          word_on_sent_contribution += copy_ptdw(i, t) / sen_subj[t];
        }
      }
      float sum = 0.0f;
      float non_backs = 1.0f;

      for (int t = 0; t < num_topics; t++) {
        if (copy_ptdw(i, t) != 0 && norm_sum != 0) {
          if (!is_background_topic[t]) {
            (*ptdw)(i, t) = copy_ptdw(i, t) * (1.0f - tau * (weight / norm_sum) *
              (1.0f / sen_subj[t] - word_on_sent_contribution));
          }
        }
        if ((*ptdw)(i, t) < 0.0f) {
          (*ptdw)(i, t) = 0.0f;
        }
        if (!is_background_topic[t]) {
          sum += (*ptdw)(i, t);
        } else {
          non_backs -= (*ptdw)(i, t);
        }
      }
      for (int t = 0; t < num_topics; t++) {
        if ((*ptdw)(i, t) != 0 && !is_background_topic[t]) {
          (*ptdw)(i, t) = non_backs * (*ptdw)(i, t) / sum;
        }
      }
    }
    if (!handling_first_sen) {
      float norm_prev = 0.0f;
      float norm_cur = 0.0f;
      float dot = 0.0f;
      for (int t = 0; t < num_topics; t++) {
        float sen_subj_t = sen_subj[t];
        float prev_sen_subj_t = prev_sen_subject[t];

        dot += prev_sen_subj_t * sen_subj_t;
        norm_cur += sen_subj_t * sen_subj_t;
        norm_prev += prev_sen_subj_t * prev_sen_subj_t;
      }
      norm_prev = std::sqrt(norm_prev);
      norm_cur = std::sqrt(norm_cur);
      float dist = 1.0f;
      if (norm_prev > 1e-5f && norm_cur > 1e-5f) {
        dist = dot / (norm_prev * norm_cur);
      }
      distances.push_back(dist);
    }
    sen_begin = sen_end + 1;
    prev_sen_subject = sen_subj;
    prev_seg_len = sen_len;
    it++;
    handling_first_sen = false;
  }
  if (merge_into_segments) {
    float mean_dist = 0.0f;
    for (int i = 0; i < distances.size(); i++) {
      mean_dist += distances[i];
    }
    mean_dist /= static_cast<int>(distances.size());
    float sigma = 0.0f;
    for (int i = 0; i < distances.size(); i++) {
      sigma += (distances[i] - mean_dist) * (distances[i] - mean_dist);
    }
    sigma /= static_cast<int>(distances.size());
    int i = 0;
    std::list<int>::iterator it = dot_positions.begin();
    int count = 0;
    for (; it != dot_positions.end(); it++) {
      if (i >= static_cast<int>(distances.size())) {
        break;
      }
      if (distances[i] > mean_dist + alpha * sigma) {
        it = dot_positions.erase(it);
        count++;
        it--;
      }
      i++;
    }
  }
}


std::shared_ptr<RegularizePtdwAgent>
TopicSegmentationPtdw::CreateRegularizePtdwAgent(const Batch& batch,
                                                 const ProcessBatchesArgs& args, float tau) {
  std::vector<std::list<int>> dot_positions;
  int dot_count = 0;
  for (int item_index = 0; item_index < batch.item_size(); item_index++) {
    std::list<int> current_dots;
    const Item& item = batch.item(item_index);
    for (int token_index = 0; token_index < item.token_id_size(); token_index++) {
      int token_id = item.token_id(token_index);
      if (batch.token(token_id) == ".") {
        current_dots.push_back(token_index);
        dot_count++;
      }
    }
    current_dots.push_back(item.token_id_size());
    dot_positions.push_back(current_dots);
  }
  LOG(INFO) << "Dot count: " << dot_count;

  TopicSegmentationPtdwAgent* agent = new TopicSegmentationPtdwAgent(config_, args, tau, dot_positions);
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
