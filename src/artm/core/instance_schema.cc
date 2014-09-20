// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/instance_schema.h"

#include "artm/regularizer_interface.h"
#include "artm/score_calculator_interface.h"
#include "artm/messages.pb.h"

namespace artm {
namespace core {

InstanceSchema::InstanceSchema()
    : config_(),  regularizers_(), models_config_(), score_calculators_() {}

InstanceSchema::InstanceSchema(const InstanceSchema& schema)
    : config_(schema.config_),
      regularizers_(schema.regularizers_),
      models_config_(schema.models_config_),
      score_calculators_(schema.score_calculators_) {}


InstanceSchema::InstanceSchema(const MasterComponentConfig& config)
    : config_(config), regularizers_(), models_config_(), score_calculators_() {}

void InstanceSchema::set_config(const MasterComponentConfig& config) {
  config_.CopyFrom(config);
}

const MasterComponentConfig& InstanceSchema::config() const {
  return config_;
}

void InstanceSchema::set_model_config(
    ModelName id, const std::shared_ptr<const ModelConfig>& model_config) {
  auto iter = models_config_.find(id);
  if (iter != models_config_.end()) {
    iter->second = model_config;
  } else {
    models_config_.insert(std::make_pair(id, model_config));
  }
}

const ModelConfig& InstanceSchema::model_config(ModelName id) const {
  auto iter = models_config_.find(id);
  return *(iter->second);
}

bool InstanceSchema::has_model_config(ModelName id) const {
  auto iter = models_config_.find(id);
  return iter != models_config_.end();
}

void InstanceSchema::clear_model_config(ModelName id) {
  auto iter = models_config_.find(id);
  if (iter != models_config_.end()) {
    models_config_.erase(iter);
  }
}

void InstanceSchema::set_regularizer(const std::string& name,
                                     const std::shared_ptr<RegularizerInterface>& regularizer) {
  auto iter = regularizers_.find(name);
  if (iter != regularizers_.end()) {
    iter->second = regularizer;
  } else {
    regularizers_.insert(std::make_pair(name, regularizer));
  }
}

bool InstanceSchema::has_regularizer(const std::string& name) const {
  auto iter = regularizers_.find(name);
  return iter != regularizers_.end();
}

void InstanceSchema::clear_regularizer(const std::string& name) {
  auto iter = regularizers_.find(name);
  if (iter != regularizers_.end()) {
    regularizers_.erase(iter);
  }
}

std::shared_ptr<RegularizerInterface> InstanceSchema::regularizer(const std::string& name) {
  auto iter = regularizers_.find(name);
  if (iter != regularizers_.end()) {
    return iter->second;
  } else {
    return nullptr;
  }
}

void InstanceSchema::set_score_calculator(
    const ScoreName& name,
    const std::shared_ptr<ScoreCalculatorInterface>& score_calculator) {
  auto iter = score_calculators_.find(name);
  if (iter != score_calculators_.end()) {
    iter->second = score_calculator;
  } else {
    score_calculators_.insert(std::make_pair(name, score_calculator));
  }
}

bool InstanceSchema::has_score_calculator(const ScoreName& name) const {
  auto iter = score_calculators_.find(name);
  return iter != score_calculators_.end();
}

void InstanceSchema::clear_score_calculator(const ScoreName& name) {
  auto iter = score_calculators_.find(name);
  if (iter != score_calculators_.end()) {
    score_calculators_.erase(iter);
  }
}

std::shared_ptr<ScoreCalculatorInterface> InstanceSchema::score_calculator(
    const ScoreName& name) {
  auto iter = score_calculators_.find(name);
  if (iter != score_calculators_.end()) {
    return iter->second;
  } else {
    return nullptr;
  }
}

void InstanceSchema::clear_score_calculators() {
  score_calculators_.clear();
}

std::vector<ModelName> InstanceSchema::GetModelNames() const {
  std::vector<ModelName> retval;
  for (auto iter = models_config_.begin(); iter != models_config_.end(); ++iter) {
    retval.push_back(iter->first);
  }

  return retval;
}

}  // namespace core
}  // namespace artm
