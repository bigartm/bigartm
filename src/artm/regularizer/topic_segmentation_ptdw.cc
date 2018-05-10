// Copyright 2017, Additive Regularization of Topic Models.

// Author: Anastasia Bayandina

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
  int local_token_size = ptdw->no_rows();
  int num_topics = ptdw->no_columns();
  std::vector<float> background_probability(local_token_size, 0.0f);
  //// if background topics are given, count probability for each word to be background
  std::vector<bool> is_background_topic = core::is_member(args_.topic_name(), config_.background_topic_names());
  if (config_.background_topic_names().size()) {
    //std::vector<bool> is_background_topic = core::is_member(args_.topic_name(), config_.background_topic_names());
    for (int i = 0; i < local_token_size; ++i) {
      const float* local_ptdw_ptr = &(*ptdw)(i, 0);  // NOLINT
      for (int k = 0; k < num_topics; ++k) {
        if (is_background_topic[k]) {
          background_probability[i] += local_ptdw_ptr[k];
        }
      }
    }
  }
  //for (int i = 0; i < local_token_size; ++i) {
  //    if (background_probability[i] > 0.99f)
  //      LOG(INFO) << i << ' ' << background_probability[i];
  //}

  //int h = config_.window();
  //double threshold_topic_change = config_.threshold();
  ::artm::utility::DenseMatrix<float> copy_ptdw(*ptdw);
  //LOG(INFO) << "Started " << inner_iter;
  //std::vector<double> left_distribution(num_topics, 0.0);
  //std::vector<double> right_distribution(num_topics, 0.0);
  //double left_weights = 0.0;
  //double right_weights = 0.0;
  //int l_topic, r_topic;  // topic ids on which maximum of the distribution is reached
  //bool changes_topic = false;
  //int main_topic = 0;
  //auto main_topic_density = (*ptdw)(0, 0);
  float tau = tau_;
  float alpha = config_.merge_threshold();
  bool merge_into_segments = config_.merge_into_segments();
  std::list<int> &dot_positions = dot_positions_[item_index];
  float average_len = 0.0f;
  int seg_begin = 0;
  for (auto it = dot_positions.begin(); it != dot_positions.end(); it++) {
    average_len += *it - seg_begin;
    seg_begin = *it + 1;
  }
  average_len /= dot_positions.size();
  //LOG(INFO) << dot_positions[dot_positions.size() - 1] << ' ' << local_token_size;
  int c = 1;
  if (inner_iter == 50) {
    c = 1;
  }
  for (int k = 0; k < c; k++){
  int sen_begin = 0;
  std::list<int>::iterator it = dot_positions.begin();
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
      //LOG(INFO) << weight << ' '<< background_probability[i];
      //weight = 1.0f;
      if (weight == 0.0f) continue;
      norm_sum += weight * weight;
      for (int t = 0; t < num_topics; t++) {
        if (is_background_topic[t])
          continue;
        sen_subj[t] += copy_ptdw(i, t) * weight;
      }
    }
    for (int t = 0; t < num_topics; t++) {
      if (is_background_topic[t])
        continue;
      if (norm_sum != 0)
        sen_subj[t] /= norm_sum;
    }
    for (int i = sen_begin; i < sen_end; i++) {
      float TMP = 0.0f;
      float weight = 1.0f - background_probability[i];
      //weight = 1.0f;
      for (int t = 0; t < num_topics; t++) {
        if (is_background_topic[t])
          continue;
        if (copy_ptdw(i, t) != 0 && sen_subj[t] != 0)
          TMP += copy_ptdw(i, t) / sen_subj[t];
      }
      float sum = 0.0f;
      float non_backs = 1.0f;
      
      for (int t = 0; t < num_topics; t++) {
        if (copy_ptdw(i, t) != 0 && norm_sum != 0) {
          if (is_background_topic[t]) {
            //(*ptdw)(i, t) = copy_ptdw(i, t) * (1.0f + tau * weight * TMP);
          } else {
           (*ptdw)(i, t) = copy_ptdw(i, t) * (1.0f - tau * (weight / norm_sum) * (1.0f / sen_subj[t] - TMP));
          }
        }
        if ((*ptdw)(i, t) < 0.0f)
          (*ptdw)(i, t) = 0.0f;
        if (! is_background_topic[t])
          sum += (*ptdw)(i, t);
        else
          non_backs -= (*ptdw)(i, t);

      }
      for (int t = 0; t < num_topics; t++) {
        if ((*ptdw)(i, t) != 0 && !is_background_topic[t])
          (*ptdw)(i, t) = non_backs * (*ptdw)(i, t) / sum;

      }
    }
    if (!handling_first_sen) {
      float norm_prev = 0.0f;
      float norm_cur = 0.0f;
      float dot = 0.0f;
      for (int t = 0; t < num_topics; t++) {
        float sen_subj_t = sen_subj[t];
        float prev_sen_subj_t = prev_sen_subject[t];
        // if (!handling_first_sen && sen_len < average_len / 2.0)
        //   sen_subj_t = (1.0f - alpha) * sen_subj[t] + alpha * prev_sen_subject[t];
        // else if (!handling_first_sen && prev_seg_len < average_len / 2.0)
        //   prev_sen_subj_t = (1.0f - alpha) * prev_sen_subject[t] + alpha * sen_subj[t];
        
        dot += prev_sen_subj_t * sen_subj_t;
        norm_cur += sen_subj_t * sen_subj_t;
        norm_prev += prev_sen_subj_t * prev_sen_subj_t;
      }
      norm_prev = std::sqrt(norm_prev);
      norm_cur = std::sqrt(norm_cur);
      float dist = 1.0f;
      if (norm_prev > 1e-5 && norm_cur > 1e-5) {
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
    for (int i = 0; i < (int)distances.size(); i++) {
      mean_dist += distances[i];
    }
    mean_dist /= (int)distances.size();
    float sigma = 0.0f;
    for (int i = 0; i < (int)distances.size(); i++) {
      sigma += (distances[i] - mean_dist) * (distances[i] - mean_dist);
    }
    sigma /= (int)distances.size();
    int i = 0;
    std::list<int>::iterator it = dot_positions.begin();
    int count = 0;
    for (; it != dot_positions.end(); it++) {
      if (i >= (int)distances.size())
        break;
      if(distances[i] > mean_dist + alpha * sigma)
      {
        it = dot_positions.erase(it);
        count++;
        it--;
      }
      i++;
    }
    // LOG(INFO) << "### " << inner_iter << ' ' << count;
  }
  }
}



//  for (int k = 0; k < num_topics; ++k) {
//    auto cur_topic_density = (*ptdw)(0, k);
//    if (main_topic_density < cur_topic_density) {
//      main_topic_density = cur_topic_density;
//      main_topic = k;
//    }
//  }
//  for (int k = 0; k < num_topics; ++k) {
//    if (k == main_topic) {
//      (*ptdw)(0, k) = 1;
//    } else {
//      (*ptdw)(0, k) = 0;
//    }
//  }
//  for (int i = 0; i < h && i < local_token_size; ++i) {
//    for (int k = 0; k < num_topics; ++k) {
//      right_distribution[k] += copy_ptdw(i, k) * (1 - background_probability[i]);
//    }
//    right_weights += 1 - background_probability[i];
//  }
//  for (int i = 1; i < local_token_size; ++i) {
//    for (int k = 0; k < num_topics; ++k) {
//      left_distribution[k] += copy_ptdw(i - 1, k) * (1 - background_probability[i - 1]);
//      right_distribution[k] -= copy_ptdw(i - 1, k) * (1 - background_probability[i - 1]);
//    }
//    left_weights += 1 - background_probability[i - 1];
//    right_weights -= 1 - background_probability[i - 1];
//    if (i <= local_token_size - h) {
//      for (int k = 0; k < num_topics; ++k) {
//        right_distribution[k] += copy_ptdw(i + h - 1, k) * (1 - background_probability[i + h - 1]);
//      }
//      right_weights += 1 - background_probability[i + h - 1];
//    }
//    if (i > h) {
//      for (int k = 0; k < num_topics; ++k) {
//        left_distribution[k] -= copy_ptdw(i - h - 1, k) * (1 - background_probability[i - h - 1]);
//      }
//      left_weights -= 1 - background_probability[i - h - 1];
//    }
//    auto lb = left_distribution.begin();
//    auto le = left_distribution.end();
//    auto rb = right_distribution.begin();
//    auto re = right_distribution.end();
//    l_topic = std::distance(lb, std::max_element(lb, le));
//    r_topic = std::distance(rb, std::max_element(rb, re));
//    double ll =  left_distribution[l_topic];
//    double rl = right_distribution[l_topic];
//    double rr = right_distribution[r_topic];
//    double lr =  left_distribution[r_topic];
//    changes_topic = ((ll / left_weights  - rl / right_weights) / 2 +
//                     (rr / right_weights - lr / left_weights)  / 2 > threshold_topic_change);
//    if (changes_topic) {
//      main_topic = r_topic;
//    }
//    for (int k = 0; k < num_topics; ++k) {
//      if (k == main_topic) {
//        (*ptdw)(i, k) = 1;
//      } else {
//        (*ptdw)(i, k) = 0;
//      }
//    }
//  }
//}

std::shared_ptr<RegularizePtdwAgent>
TopicSegmentationPtdw::CreateRegularizePtdwAgent(const Batch& batch,
                                                 const ProcessBatchesArgs& args, double tau) {
  std::vector< std::list<int> > dot_positions;
  int dot_count = 0;
  for (int item_index = 0; item_index < batch.item_size(); item_index++) {
    std::list<int> current_dots;
    const Item& item = batch.item(item_index);
    for (int token_index = 0; token_index < item.token_id_size(); token_index++) {
      int token_id = item.token_id(token_index);
      if (batch.token(token_id) == ".") {
        current_dots.push_back(token_index);
        dot_count += 1;
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
