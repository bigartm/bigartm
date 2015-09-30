// Copyright 2014, Additive Regularization of Topic Models.

#include <memory>
#include <string>
#include <utility>

#include "artm/core/instance.h"

#include "boost/bind.hpp"

#include "artm/core/common.h"
#include "artm/core/helpers.h"
#include "artm/core/data_loader.h"
#include "artm/core/batch_manager.h"
#include "artm/core/cache_manager.h"
#include "artm/core/dictionary.h"
#include "artm/core/exceptions.h"
#include "artm/core/processor.h"
#include "artm/core/merger.h"
#include "artm/core/template_manager.h"
#include "artm/core/topic_model.h"
#include "artm/core/instance_schema.h"

#include "artm/regularizer_interface.h"
#include "artm/regularizer/decorrelator_phi.h"
#include "artm/regularizer/multilanguage_phi.h"
#include "artm/regularizer/smooth_sparse_theta.h"
#include "artm/regularizer/smooth_sparse_phi.h"
#include "artm/regularizer/label_regularization_phi.h"
#include "artm/regularizer/specified_sparse_phi.h"
#include "artm/regularizer/improve_coherence_phi.h"

#include "artm/score/items_processed.h"
#include "artm/score/sparsity_theta.h"
#include "artm/score/sparsity_phi.h"
#include "artm/score/top_tokens.h"
#include "artm/score/topic_kernel.h"
#include "artm/score/theta_snippet.h"
#include "artm/score/perplexity.h"
#include "artm/score/topic_mass_phi.h"

#define CREATE_OR_RECONFIGURE_REGULARIZER(ConfigType, RegularizerType) {                      \
  ConfigType regularizer_config;                                                              \
  if (!regularizer_config.ParseFromArray(config_blob.c_str(), config_blob.length())) {        \
    BOOST_THROW_EXCEPTION(CorruptedMessageException("Unable to parse regularizer config"));   \
  }                                                                                           \
  if (need_hot_reconfigure) {                                                                 \
    need_hot_reconfigure = regularizer->Reconfigure(config);                                  \
  }                                                                                           \
  if (!need_hot_reconfigure) {                                                                \
    regularizer.reset(new RegularizerType(regularizer_config));                               \
    LOG(INFO) << "Regularizer '" + regularizer_name + "' was cold-reconfigured";              \
  } else {                                                                                    \
    LOG(INFO) << "Regularizer '" + regularizer_name + "' was hot-reconfigured";               \
  }                                                                                           \
}                                                                                             \

#define CREATE_SCORE_CALCULATOR(ConfigType, ScoreType) {                                  \
  ConfigType score_config;                                                                \
  if (!score_config.ParseFromArray(config_blob.c_str(), config_blob.length())) {          \
    BOOST_THROW_EXCEPTION(CorruptedMessageException("Unable to parse score config"));     \
  }                                                                                       \
  score_calculator.reset(new ScoreType(score_config));                                    \
}                                                                                         \

namespace artm {
namespace core {

Instance::Instance(const MasterComponentConfig& config)
    : is_configured_(false),
      schema_(std::make_shared<InstanceSchema>(config)),
      dictionaries_(),
      batches_(),
      processor_queue_(),
      merger_queue_(),
      cache_manager_(),
      batch_manager_(),
      data_loader_(nullptr),
      merger_(),
      processors_() {
  Reconfigure(config);
}

Instance::Instance(const Instance& rhs)
    : is_configured_(false),
      schema_(rhs.schema()->Duplicate()),
      dictionaries_(),
      batches_(),
      processor_queue_(),
      merger_queue_(),
      cache_manager_(),
      batch_manager_(),
      data_loader_(nullptr),
      merger_(),
      processors_() {
  Reconfigure(schema_.get()->config());

  std::vector<std::string> dict_name = rhs.dictionaries_.keys();
  for (auto& key : dict_name) {
    std::shared_ptr<Dictionary> value = rhs.dictionaries_.get(key);
    if (value != nullptr)
      dictionaries_.set(key, value->Duplicate());
  }

  std::vector<std::string> batch_name = rhs.batches_.keys();
  for (auto& key : batch_name) {
    std::shared_ptr<Batch> value = rhs.batches_.get(key);
    if (value != nullptr)
      batches_.set(key, value);  // store same batch as rhs (OK as batches here are read-only)
  }

  std::vector<ModelName> model_name = rhs.merger_->model_name();
  for (auto& key : model_name) {
    std::shared_ptr<const PhiMatrix> value = rhs.merger_->GetPhiMatrix(key);
    if (value != nullptr)
      merger_->SetPhiMatrix(key, value->Duplicate());
  }
}

Instance::~Instance() {}

std::shared_ptr<Instance> Instance::Duplicate() const {
  return std::shared_ptr<Instance>(new Instance(*this));
}

void Instance::RequestMasterComponentInfo(MasterComponentInfo* master_info) const {
  schema_.get()->RequestMasterComponentInfo(master_info);
  cache_manager_->RequestMasterComponentInfo(master_info);

  for (auto& name : dictionaries_.keys()) {
    std::shared_ptr<Dictionary> dict = dictionaries_.get(name);
    if (dict == nullptr)
      continue;

    MasterComponentInfo::DictionaryInfo* info = master_info->add_dictionary();
    info->set_name(name);
    info->set_entries_count(dict->size());
  }

  for (auto& name : batches_.keys()) {
    std::shared_ptr<Batch> batch = batches_.get(name);
    if (batch == nullptr)
      continue;

    MasterComponentInfo::BatchInfo* info = master_info->add_batch();
    info->set_name(name);
    info->set_token_count(batch->token_size());
    info->set_items_count(batch->item_size());
  }

  for (auto& name : merger_->model_name()) {
    std::shared_ptr<const PhiMatrix> phi_matrix = merger_->GetPhiMatrix(name);
    std::shared_ptr<const TopicModel> topic_model = merger_->GetLatestTopicModel(name);
    if (phi_matrix == nullptr && topic_model == nullptr)
      continue;
    const PhiMatrix& p_wt = topic_model != nullptr ? topic_model->GetPwt() : *phi_matrix;

    MasterComponentInfo::ModelInfo* info = master_info->add_model();
    info->set_name(p_wt.model_name());
    info->set_token_count(p_wt.token_size());
    info->set_topics_count(p_wt.topic_size());
    info->set_type(topic_model != nullptr ? typeid(*topic_model).name() : typeid(*phi_matrix).name());
  }

  master_info->set_merger_queue_size(merger_queue_.size());
  master_info->set_processor_queue_size(processor_queue_.size());
}

DataLoader* Instance::data_loader() {
  return data_loader_.get();
}

BatchManager* Instance::batch_manager() {
  return batch_manager_.get();
}

CacheManager* Instance::cache_manager() {
  return cache_manager_.get();
}

Merger* Instance::merger() {
  return merger_.get();
}

void Instance::CreateOrReconfigureModel(const ModelConfig& config) {
  if (!Helpers::Validate(config)) return;
  auto corrected_config = std::make_shared<artm::ModelConfig>(config);
  if (merger_ != nullptr) {
    merger_->CreateOrReconfigureModel(*corrected_config);
  }

  auto new_schema = schema_.get_copy();
  auto const_config = std::const_pointer_cast<const ModelConfig>(corrected_config);
  new_schema->set_model_config(const_config->name(), const_config);
  schema_.set(new_schema);
}

void Instance::DisposeModel(ModelName model_name) {
  auto new_schema = schema_.get_copy();
  new_schema->clear_model_config(model_name);
  schema_.set(new_schema);

  if (merger_ != nullptr) {
    merger_->DisposeModel(model_name);
  }

  if (batch_manager_ != nullptr) {
    batch_manager_->DisposeModel(model_name);
  }

  if (cache_manager_ != nullptr) {
    cache_manager_->DisposeModel(model_name);
  }
}

void Instance::CreateOrReconfigureRegularizer(const RegularizerConfig& config) {
  std::string regularizer_name = config.name();
  artm::RegularizerConfig_Type regularizer_type = config.type();

  std::shared_ptr<artm::RegularizerInterface> regularizer;
  bool need_hot_reconfigure = schema_.get()->has_regularizer(regularizer_name);
  if (need_hot_reconfigure) {
    regularizer = schema_.get()->regularizer(regularizer_name);
  } else {
    LOG(INFO) << "Regularizer '" + regularizer_name + "' will be created";
  }

  std::string config_blob;  // Used by CREATE_OR_RECONFIGURE_REGULARIZER marco
  if (config.has_config()) {
    config_blob = config.config();
  }

  // add here new case if adding new regularizer
  switch (regularizer_type) {
    case artm::RegularizerConfig_Type_SmoothSparseTheta: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::SmoothSparseThetaConfig,
                                        ::artm::regularizer::SmoothSparseTheta);
      break;
    }

    case artm::RegularizerConfig_Type_SmoothSparsePhi: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::SmoothSparsePhiConfig,
                                        ::artm::regularizer::SmoothSparsePhi);
      break;
    }

    case artm::RegularizerConfig_Type_LabelRegularizationPhi: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::LabelRegularizationPhiConfig,
                                        ::artm::regularizer::LabelRegularizationPhi);
      break;
    }

    case artm::RegularizerConfig_Type_DecorrelatorPhi: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::DecorrelatorPhiConfig,
                                        ::artm::regularizer::DecorrelatorPhi);
      break;
    }

    case artm::RegularizerConfig_Type_MultiLanguagePhi: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::MultiLanguagePhiConfig,
                                        ::artm::regularizer::MultiLanguagePhi);
      break;
    }

    case artm::RegularizerConfig_Type_SpecifiedSparsePhi: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::SpecifiedSparsePhiConfig,
                                        ::artm::regularizer::SpecifiedSparsePhi);
      break;
    }

    case artm::RegularizerConfig_Type_ImproveCoherencePhi: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::ImproveCoherencePhiConfig,
                                        ::artm::regularizer::ImproveCoherencePhi);
      break;
    }

    default:
      BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
        "RegularizerConfig.type", regularizer_type));
  }

  regularizer->set_dictionaries(&dictionaries_);
  auto new_schema = schema_.get_copy();
  new_schema->set_regularizer(regularizer_name, regularizer);
  schema_.set(new_schema);
}

std::shared_ptr<ScoreCalculatorInterface> Instance::CreateScoreCalculator(const ScoreConfig& config) {
  std::string score_name = config.name();
  artm::ScoreConfig_Type score_type = config.type();

  std::string config_blob;  // Used by CREATE_SCORE_CALCULATOR macro
  if (config.has_config()) {
    config_blob = config.config();
  }

  std::shared_ptr<artm::ScoreCalculatorInterface> score_calculator;

  // add here new case if adding new score
  switch (score_type) {
    case artm::ScoreConfig_Type_Perplexity: {
      CREATE_SCORE_CALCULATOR(::artm::PerplexityScoreConfig,
                              ::artm::score::Perplexity);
      break;
    }

    case artm::ScoreConfig_Type_SparsityTheta: {
      CREATE_SCORE_CALCULATOR(::artm::SparsityThetaScoreConfig,
                              ::artm::score::SparsityTheta);
      break;
    }

    case artm::ScoreConfig_Type_SparsityPhi: {
      CREATE_SCORE_CALCULATOR(::artm::SparsityPhiScoreConfig,
                              ::artm::score::SparsityPhi);
      break;
    }

    case artm::ScoreConfig_Type_ItemsProcessed: {
      CREATE_SCORE_CALCULATOR(::artm::ItemsProcessedScoreConfig,
                              ::artm::score::ItemsProcessed);
      break;
    }

    case artm::ScoreConfig_Type_TopTokens: {
      CREATE_SCORE_CALCULATOR(::artm::TopTokensScoreConfig,
                              ::artm::score::TopTokens);
      break;
    }

    case artm::ScoreConfig_Type_ThetaSnippet: {
      CREATE_SCORE_CALCULATOR(::artm::ThetaSnippetScoreConfig,
                              ::artm::score::ThetaSnippet);
      break;
    }

    case artm::ScoreConfig_Type_TopicKernel: {
      CREATE_SCORE_CALCULATOR(::artm::TopicKernelScoreConfig,
                              ::artm::score::TopicKernel);
      break;
    }

    case artm::ScoreConfig_Type_TopicMassPhi: {
      CREATE_SCORE_CALCULATOR(::artm::TopicMassPhiScoreConfig,
                              ::artm::score::TopicMassPhi);
      break;
    }

    default:
      BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException("ScoreConfig.type", score_type));
  }
  score_calculator->set_dictionaries(&dictionaries_);
  return score_calculator;
}

void Instance::DisposeRegularizer(const std::string& name) {
  auto new_schema = schema_.get_copy();
  new_schema->clear_regularizer(name);
  schema_.set(new_schema);
}

void Instance::CreateOrReconfigureDictionary(const DictionaryConfig& config) {
  auto dictionary = std::make_shared<Dictionary>(Dictionary(config));
  dictionaries_.set(config.name(), dictionary);
}

void Instance::DisposeDictionary(const std::string& name) {
  dictionaries_.erase(name);
}

void Instance::Reconfigure(const MasterComponentConfig& master_config) {
  MasterComponentConfig old_config = schema_.get()->config();

  auto new_schema = schema_.get_copy();
  new_schema->set_config(master_config);

  new_schema->clear_score_calculators();  // Clear all score calculators
  for (int score_index = 0;
       score_index < master_config.score_config_size();
       ++score_index) {
    const ScoreConfig& score_config = master_config.score_config(score_index);
    auto score_calculator = CreateScoreCalculator(score_config);
    new_schema->set_score_calculator(score_config.name(), score_calculator);
  }

  schema_.set(new_schema);

  if (!is_configured_) {
    // First reconfiguration.
    cache_manager_.reset(new CacheManager());
    batch_manager_.reset(new BatchManager());
    data_loader_.reset(new DataLoader(this));
    merger_.reset(new Merger(&merger_queue_, &schema_, &batches_, &dictionaries_));

    is_configured_  = true;
  }

  {
    // Adjust size of processors_; cast size to int to avoid compiler warning.
    while (static_cast<int>(processors_.size()) > master_config.processors_count()) {
      processors_.pop_back();
    }

    while (static_cast<int>(processors_.size()) < master_config.processors_count()) {
      processors_.push_back(
        std::shared_ptr<Processor>(new Processor(
          &processor_queue_,
          &merger_queue_,
          batches_,
          *merger_,
          schema_)));
    }
  }
}

}  // namespace core
}  // namespace artm
