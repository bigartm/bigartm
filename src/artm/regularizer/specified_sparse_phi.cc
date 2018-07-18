// Copyright 2017, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <string>
#include <vector>
#include <queue>
#include <utility>

#include "artm/core/protobuf_helpers.h"
#include "artm/core/phi_matrix.h"
#include "artm/regularizer/specified_sparse_phi.h"

namespace artm {
namespace regularizer {

struct Comparator {
  bool operator() (std::pair<int, float> left, std::pair<int, float> right) {
    return left.second > right.second;
  }
};

bool SpecifiedSparsePhi::RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                                       const ::artm::core::PhiMatrix& n_wt,
                                       ::artm::core::PhiMatrix* result) {
  // read the parameters from config and control their correctness
  const int topic_size = n_wt.topic_size();
  const int token_size = n_wt.token_size();

  std::vector<bool> topics_to_regularize;
  if (config_.topic_name().size() == 0) {
    topics_to_regularize.assign(topic_size, true);
  } else {
    topics_to_regularize = core::is_member(n_wt.topic_name(), config_.topic_name());
  }

  const  bool mode_topics = config_.mode() == artm::SpecifiedSparsePhiConfig_SparseMode_SparseTopics;
  const int global_end = mode_topics ? topic_size : token_size;
  const int local_end = !mode_topics ? topic_size : token_size;

  const auto& class_id = config_.class_id();

  // proceed the regularization
  for (int global_index = 0; global_index < global_end; ++global_index) {
    if (mode_topics) {
      if (!topics_to_regularize[global_index]) {
        continue;
      }
    } else {
      const auto& token = n_wt.token(global_index);
      if (token.class_id != class_id) {
        continue;
      }
    }

    google::protobuf::RepeatedField<int> indices_of_max;
    std::vector<std::pair<int, float> > max_and_indices;
    std::priority_queue<std::pair<int, float>,
                        std::vector<std::pair<int, float>>,
                        Comparator> max_queue;
    double normalizer = 0.0;

    for (int local_index = 0; local_index < local_end; ++local_index) {
      if (mode_topics) {
        const auto& token = n_wt.token(local_index);
        if (token.class_id != class_id) {
          continue;
        }
      } else {
        if (!topics_to_regularize[local_index]) {
          continue;
        }
      }

      const auto value = std::pair<int, float>(local_index,
          mode_topics ? n_wt.get(local_index, global_index) : n_wt.get(global_index, local_index));
      normalizer += value.second;

      if (max_queue.size() < config_.max_elements_count()) {
        max_queue.push(value);
        continue;
      }
      if (value > max_queue.top()) {
        max_queue.pop();
        max_queue.push(value);
      }
    }

    // get maxes and their indices
    int max_queue_size = static_cast<int>(max_queue.size());
    for (int i = 0; i < max_queue_size; ++i) {
      max_and_indices.push_back(max_queue.top());
      max_queue.pop();
    }

    // check the threshold
    int stop_index = 0;
    double sum = 0.0;
    for (int i = max_and_indices.size() - 1; i >= 0; --i) {
      sum += max_and_indices[i].second;
      if (sum / normalizer >= config_.probability_threshold()) {
        stop_index = i;
        break;
      }
    }
    for (unsigned i = stop_index; i < max_and_indices.size(); ++i) {
      int* ptr = indices_of_max.Add();
      *ptr = max_and_indices[i].first;
    }

    // apply additions
    google::protobuf::RepeatedField<int> all_indices;
    for (int i = 0; i < local_end; ++i) {
      int* ptr = all_indices.Add();
      *ptr = i;
    }
    auto saved_indices = artm::core::is_member(all_indices, indices_of_max);
    for (int local_index = 0; local_index < local_end; ++local_index) {
      if (mode_topics) {
        result->set(local_index, global_index,
                    saved_indices[local_index] ? 0.0f : -n_wt.get(local_index, global_index));
      } else {
        result->set(global_index, local_index,
                    saved_indices[local_index] ? 0.0f : -n_wt.get(global_index, local_index));
      }
    }
  }
  return true;
}

google::protobuf::RepeatedPtrField<std::string> SpecifiedSparsePhi::topics_to_regularize() {
  return config_.topic_name();
}

google::protobuf::RepeatedPtrField<std::string> SpecifiedSparsePhi::class_ids_to_regularize() {
  google::protobuf::RepeatedPtrField<std::string> retval;
  std::string* ptr = retval.Add();
  *ptr = config_.class_id();
  return retval;
}

bool SpecifiedSparsePhi::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  SpecifiedSparsePhiConfig regularizer_config;
  if (!regularizer_config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse SpecifiedSparsePhiConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  return true;
}

}  // namespace regularizer
}  // namespace artm
