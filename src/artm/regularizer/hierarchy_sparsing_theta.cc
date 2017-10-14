// Copyright 2017, Additive Regularization of Topic Models.

// Author: Nadia Chirkova (nadiinchi@gmail.com)
// Based on code of Murat Apishev (great-mel@yandex.ru)

#include <vector>

#include "glog/logging.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/regularizer/hierarchy_sparsing_theta.h"

namespace artm {
namespace regularizer {

void HierarchySparsingThetaAgent::Apply(int item_index, int inner_iter, int topics_size,
                                        const float* n_td, float* r_td) const {
  LOG_FIRST_N(ERROR, 1) << "HierarchySparsingTheta regularizer can not be applied with opt_for_avx=False. "
                        << "Regularization will be skipped.";
}

void HierarchySparsingThetaAgent::Apply(int inner_iter,
                                        const ::artm::utility::LocalThetaMatrix<float>& n_td,
                                        ::artm::utility::LocalThetaMatrix<float>* r_td) const {
  if (!(regularization_on)) {
    return;
  }

  std::vector<float> n_d, n_t;
  int topic_size = n_td.num_topics();
  int item_size = n_td.num_items();
  float item_sum = 0.0f;
  float topic_sum = 0.0f;

  // count n_d
  for (int item_id = 0; item_id < item_size; ++item_id) {
    item_sum = 0;
    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      item_sum += n_td(topic_id, item_id);
    }
    n_d.push_back(item_sum);
  }

  // count topics proportion (n_t)
  for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
    topic_sum = 0;
    for (int item_id = 0; item_id < item_size; ++item_id) {
      topic_sum += parent_topic_proportion[item_id] * n_td(topic_id, item_id) / n_d[item_id];
    }
    n_t.push_back(topic_sum);
  }

  // proceed regularization
  if (topic_size != topic_weight.size() || inner_iter >= alpha_weight.size()) {
    return;
  }

  for (int item_id = 0; item_id < item_size; ++item_id) {
    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      if (n_td(topic_id, item_id) > 0.0f) {
        (*r_td)(topic_id, item_id) += alpha_weight[inner_iter] * topic_weight[topic_id] *
        (prior_parent_topic_probability
        - n_td(topic_id, item_id) / n_d[item_id]
        * parent_topic_proportion[item_id]
        / n_t[topic_id]);
      }
    }
  }
}

HierarchySparsingTheta::HierarchySparsingTheta(const HierarchySparsingThetaConfig& config) : config_(config) { }

std::shared_ptr<RegularizeThetaAgent>
HierarchySparsingTheta::CreateRegularizeThetaAgent(const Batch& batch,
                                                   const ProcessBatchesArgs& args, float tau) {
  HierarchySparsingThetaAgent* agent = new HierarchySparsingThetaAgent();
  std::shared_ptr<HierarchySparsingThetaAgent> retval(agent);

  const int topic_size = args.topic_name_size();
  const int item_size = batch.item_size();

  if (batch.description() != "__parent_phi_matrix_batch__") {
    agent->regularization_on = false;
    return retval;
  }

  agent->regularization_on = true;
  agent->prior_parent_topic_probability = 1 / item_size;  // each document in phi parent batch is parent topic

  if (config_.alpha_iter_size()) {
    if (args.num_document_passes() != config_.alpha_iter_size()) {
      LOG(ERROR) << "ProcessBatchesArgs.num_document_passes() != HierarchySparsingThetaConfig.alpha_iter_size()";
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

  if (config_.parent_topic_proportion_size()) {
    if (item_size != config_.parent_topic_proportion_size()) {
      LOG(ERROR) << "Batch.item_size != HierarchySparsingThetaConfig.parent_topic_proportion_size()";
      return nullptr;
    }

    for (int i = 0; i < config_.parent_topic_proportion_size(); ++i) {
      agent->parent_topic_proportion.push_back(config_.parent_topic_proportion(i));
    }
  } else {
    for (int i = 0; i < item_size; ++i) {
      agent->parent_topic_proportion.push_back(1.0f);
    }
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

google::protobuf::RepeatedPtrField<std::string> HierarchySparsingTheta::topics_to_regularize() {
  return config_.topic_name();
}

bool HierarchySparsingTheta::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  HierarchySparsingThetaConfig regularizer_config;
  if (!regularizer_config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse HierarchySparsingThetaConfig from RegularizerConfig.config"));
  }
  config_.CopyFrom(regularizer_config);

  return true;
}

}  // namespace regularizer
}  // namespace artm
