// Copyright 2017, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <vector>

#include "glog/logging.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/regularizer/topic_selection_theta.h"

namespace artm {
namespace regularizer {

void TopicSelectionThetaAgent::Apply(int item_index, int inner_iter, int topics_size,
                                     const float* n_td, float* r_td) const {
  if (topic_value.empty()) {
    LOG_FIRST_N(ERROR, 1) << "TopicSelectionThetaAgent regularizer can not be applied with opt_for_avx=False. "
                          << "Regularization will be skipped.";
    return;
  }

  assert(topics_size == topic_weight.size());
  assert(inner_iter < alpha_weight.size());

  if (topics_size != topic_weight.size() || inner_iter >= alpha_weight.size()) {
    return;
  }

  for (int topic_id = 0; topic_id < topics_size; ++topic_id) {
    if (n_td[topic_id] > 0.0f) {
      r_td[topic_id] += alpha_weight[inner_iter] * topic_weight[topic_id] * topic_value[topic_id] * n_td[topic_id];
    }
  }
}

void TopicSelectionThetaAgent::Apply(int inner_iter,
                                     const ::artm::utility::LocalThetaMatrix<float>& n_td,
                                     ::artm::utility::LocalThetaMatrix<float>* r_td) const {
  const int topic_size = n_td.num_topics();
  const int item_size = n_td.num_items();

  assert(topic_size == topic_weight.size());
  assert(inner_iter < alpha_weight.size());

  if (topic_size != topic_weight.size() || inner_iter >= alpha_weight.size()) {
    return;
  }

  auto local_topic_value = topic_value;
  if (local_topic_value.empty()) {
    std::vector<float> n_t(topic_size, 0.0f);
    double n = 0.0;

    // count n_t
    for (int item_id = 0; item_id < item_size; ++item_id) {
      for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
        n_t[topic_id] += n_td(topic_id, item_id);
      }
    }

    // count n = sum(n_t)
    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      n += n_t[topic_id];
    }

    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      float val = n_t[topic_id] * topic_size;
      local_topic_value.push_back((val > 0) ? static_cast<float>(n / val) : 0.0f);
    }
  }

  for (int item_id = 0; item_id < item_size; ++item_id) {
    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      if (n_td(topic_id, item_id) > 0.0f) {
        (*r_td)(topic_id, item_id) += alpha_weight[inner_iter] *
                                      topic_weight[topic_id] *
                                      local_topic_value[topic_id] *
                                      n_td(topic_id, item_id);
      }
    }
  }
}

TopicSelectionTheta::TopicSelectionTheta(const TopicSelectionThetaConfig& config) : config_(config) { }

std::shared_ptr<RegularizeThetaAgent>
TopicSelectionTheta::CreateRegularizeThetaAgent(const Batch& batch,
                                                const ProcessBatchesArgs& args, float tau) {
  TopicSelectionThetaAgent* agent = new TopicSelectionThetaAgent();
  std::shared_ptr<TopicSelectionThetaAgent> retval(agent);

  const int topic_size = args.topic_name_size();

  if (config_.alpha_iter_size()) {
    if (args.num_document_passes() != config_.alpha_iter_size()) {
      LOG(ERROR) << "ProcessBatchesArgs.num_document_passes() != TopicSelectionThetaConfig.alpha_iter_size()";
      return nullptr;
    }

    for (int i = 0; i < config_.alpha_iter_size(); ++i) {
      agent->alpha_weight.push_back(config_.alpha_iter(i));
    }
  } else {
    for (int i = 0; i < args.num_document_passes(); ++i) {
      agent->alpha_weight.push_back(1.0f);
    }
  }

  if (config_.topic_value_size()) {
    if (topic_size != config_.topic_value_size()) {
      LOG(ERROR) << "ProcessBatchesArgs.topic_name_size() != TopicSelectionThetaConfig.topic_value_size()";
      return nullptr;
    }

    for (int i = 0; i < topic_size; ++i) {
      agent->topic_value.push_back(config_.topic_value(i));
    }
  } else {
    // Keep topic_value empty, and assume we are running in opt_for_avx mode.
    // Then topic_value will be calculated from local theta matrix (per batch).
  }

  agent->topic_weight.resize(topic_size, 0.0f);
  if (config_.topic_name_size() == 0) {
    for (int i = 0; i < topic_size; ++i) {
      agent->topic_weight[i] = -tau;
    }
  } else {
    if (topic_size != args.topic_name_size()) {
      LOG(ERROR) << "args.num_topics() != args.topic_name_size()";
      return nullptr;
    }

    for (int topic_id = 0; topic_id < config_.topic_name_size(); ++topic_id) {
      int topic_index = ::artm::core::repeated_field_index_of(
        args.topic_name(), config_.topic_name(topic_id));
      if (topic_index != -1) {
        agent->topic_weight[topic_index] = -tau;
      }
    }
  }

  return retval;
}

google::protobuf::RepeatedPtrField<std::string> TopicSelectionTheta::topics_to_regularize() {
  return config_.topic_name();
}

bool TopicSelectionTheta::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  TopicSelectionThetaConfig regularizer_config;
  if (!regularizer_config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse TopicSelectionThetaConfig from RegularizerConfig.config"));
  }
  config_.CopyFrom(regularizer_config);

  return true;
}

}  // namespace regularizer
}  // namespace artm
