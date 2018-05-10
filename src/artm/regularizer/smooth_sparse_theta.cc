// Copyright 2017, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <vector>
#include <utility>

#include "glog/logging.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/regularizer/smooth_sparse_theta.h"

namespace artm {
namespace regularizer {

void SmoothSparseThetaAgent::Apply(int item_index, int inner_iter,
                                   int topics_size, const float* n_td, float* r_td) const {
  assert(item_index >= 0 && item_index < batch_.item_size());
  assert(topics_size == topic_weight.size());
  assert(inner_iter < alpha_weight.size());

  if (topics_size != topic_weight.size() || inner_iter >= alpha_weight.size()) {
    return;
  }

  const Item& item = batch_.item(item_index);
  const std::string& item_title = item.has_title() ? item.title() : std::string();

  bool use_specific_items = (item_topic_multiplier_ != nullptr);
  bool use_specific_multiplier = (use_specific_items &&
                                  item_topic_multiplier_->begin()->second.size() > 0);
  bool use_universal_multiplier = (!use_specific_multiplier && universal_topic_multiplier_ != nullptr);

  if (use_universal_multiplier && universal_topic_multiplier_->size() != topics_size) {
    LOG(ERROR) << "Universal topic coefs vector has length != topic_size ("
               << universal_topic_multiplier_->size() << " instead of " << topics_size << ")";;
    return;
  }

  if (use_specific_items) {
    auto iter = item_topic_multiplier_->find(item_title);
    if (item_title.empty() || iter == item_topic_multiplier_->end()) {
      return;
    }

    if (use_specific_multiplier && iter->second.size() != topics_size) {
      LOG(ERROR) << "Topic coefs vector for item " << iter->first << " has length != topic_size ("
                 << iter->second.size() << " instead of " << topics_size << ")";
      return;
    }

    for (int topic_id = 0; topic_id < topics_size; ++topic_id) {
      float mult = use_specific_multiplier ? iter->second[topic_id] :
        (use_universal_multiplier ? (*universal_topic_multiplier_)[topic_id]: 1.0);
      float value = transform_function_->apply(n_td[topic_id]);
      r_td[topic_id] += value > 0.0f ? mult * alpha_weight[inner_iter] * topic_weight[topic_id] * value : 0.0f;
    }
  } else {
    for (int topic_id = 0; topic_id < topics_size; ++topic_id) {
      float mult = use_universal_multiplier ? (*universal_topic_multiplier_)[topic_id] : 1.0;
      float value = transform_function_->apply(n_td[topic_id]);
      r_td[topic_id] += value > 0.0f ? mult * alpha_weight[inner_iter] * topic_weight[topic_id] * value : 0.0f;
    }
  }
}

SmoothSparseTheta::SmoothSparseTheta(const SmoothSparseThetaConfig& config)
    : config_(config)
    , transform_function_(nullptr)
    , item_topic_multiplier_(nullptr)
    , universal_topic_multiplier_(nullptr) {
  ReconfigureImpl();
}

std::shared_ptr<RegularizeThetaAgent>
SmoothSparseTheta::CreateRegularizeThetaAgent(const Batch& batch,
                                              const ProcessBatchesArgs& args, float tau) {
  SmoothSparseThetaAgent* agent = new SmoothSparseThetaAgent(batch,
                                                             transform_function_,
                                                             item_topic_multiplier_,
                                                             universal_topic_multiplier_);
  std::shared_ptr<SmoothSparseThetaAgent> retval(agent);

  const int topic_size = args.topic_name_size();

  if (config_.alpha_iter_size() != 0) {
    if (args.num_document_passes() != config_.alpha_iter_size()) {
      LOG(ERROR) << "ProcessBatchesArgs.num_document_passes() != SmoothSparseThetaConfig.alpha_iter_size()";
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

  agent->topic_weight.resize(topic_size, 0.0f);
  if (config_.topic_name_size() == 0) {
    for (int i = 0; i < topic_size; ++i) {
      agent->topic_weight[i] = tau;
    }
  } else {
    for (int topic_id = 0; topic_id < config_.topic_name_size(); ++topic_id) {
      int topic_index = ::artm::core::repeated_field_index_of(
        args.topic_name(), config_.topic_name(topic_id));
      if (topic_index != -1) {
        agent->topic_weight[topic_index] = tau;
      }
    }
  }

  return retval;
}

google::protobuf::RepeatedPtrField<std::string> SmoothSparseTheta::topics_to_regularize() {
  return config_.topic_name();
}

void SmoothSparseTheta::ReconfigureImpl() {
  if (config_.has_transform_config()) {
    transform_function_ = artm::core::TransformFunction::create(config_.transform_config());
  } else {
    transform_function_ = artm::core::TransformFunction::create();
  }

  if (config_.item_topic_multiplier_size() == 1) {
    universal_topic_multiplier_.reset(
      new std::vector<float>(config_.item_topic_multiplier(0).value().begin(),
                              config_.item_topic_multiplier(0).value().end()));
  }

  if (config_.item_title_size() > 0) {
    item_topic_multiplier_.reset(new ItemTopicMultiplier());
    if (config_.item_topic_multiplier_size() == config_.item_title_size()) {
      for (int i = 0; i < config_.item_title_size(); ++i) {
        item_topic_multiplier_->insert(std::make_pair(config_.item_title(i),
                                       std::vector<float>(config_.item_topic_multiplier(i).value().begin(),
                                       config_.item_topic_multiplier(i).value().end())));
        auto m_ptr = config_.mutable_item_topic_multiplier(i);
        m_ptr->clear_value();
      }
    } else {
      LOG(WARNING) << "SmoothSparseThetaConfig.item_topic_multilplier has incorrect size or is empty";
      for (int i = 0; i < config_.item_title_size(); ++i) {
        item_topic_multiplier_->insert(std::make_pair(config_.item_title(i), std::vector<float>()));
      }
    }
  }

  config_.clear_item_title();
}

bool SmoothSparseTheta::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  SmoothSparseThetaConfig regularizer_config;
  if (!regularizer_config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse SmoothSparseThetaConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  ReconfigureImpl();

  return true;
}

}  // namespace regularizer
}  // namespace artm
