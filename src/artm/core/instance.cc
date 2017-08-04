// Copyright 2017, Additive Regularization of Topic Models.

#include <memory>
#include <string>
#include <utility>
#include <thread>  // NOLINT

#include "artm/core/instance.h"

#include "boost/bind.hpp"
#include "boost/filesystem.hpp"

#include "artm/core/common.h"
#include "artm/core/helpers.h"
#include "artm/core/cache_manager.h"
#include "artm/core/score_manager.h"
#include "artm/core/dictionary.h"
#include "artm/core/exceptions.h"
#include "artm/core/processor.h"

#include "artm/regularizer_interface.h"
#include "artm/regularizer/decorrelator_phi.h"
#include "artm/regularizer/multilanguage_phi.h"
#include "artm/regularizer/smooth_sparse_theta.h"
#include "artm/regularizer/smooth_sparse_phi.h"
#include "artm/regularizer/label_regularization_phi.h"
#include "artm/regularizer/specified_sparse_phi.h"
#include "artm/regularizer/improve_coherence_phi.h"
#include "artm/regularizer/smooth_ptdw.h"
#include "artm/regularizer/topic_selection_theta.h"
#include "artm/regularizer/biterms_phi.h"
#include "artm/regularizer/hierarchy_sparsing_theta.h"
#include "artm/regularizer/topic_segmentation_ptdw.h"
#include "artm/regularizer/smooth_time_in_topics_phi.h"
#include "artm/regularizer/net_plsa_phi.h"

#include "artm/score/items_processed.h"
#include "artm/score/sparsity_theta.h"
#include "artm/score/sparsity_phi.h"
#include "artm/score/top_tokens.h"
#include "artm/score/topic_kernel.h"
#include "artm/score/theta_snippet.h"
#include "artm/score/peak_memory.h"
#include "artm/score/perplexity.h"
#include "artm/score/topic_mass_phi.h"
#include "artm/score/class_precision.h"
#include "artm/score/background_tokens_ratio.h"

#define CREATE_OR_RECONFIGURE_REGULARIZER(ConfigType, RegularizerType) {                      \
  ConfigType regularizer_config;                                                              \
  if (!regularizer_config.ParseFromString(config_blob)) {                                     \
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

namespace artm {
namespace core {

Instance::Instance(const MasterModelConfig& config)
    : is_configured_(false),
      master_model_config_(nullptr),  // copied in Reconfigure (see below)
      regularizers_(),
      score_calculators_(),
      batches_(),
      models_(),
      processor_queue_(),
      cache_manager_(),
      score_manager_(),
      score_tracker_(),
      processors_() {
  Reconfigure(config);
}

Instance::Instance(const Instance& rhs)
    : is_configured_(false),
      master_model_config_(nullptr),  // copied in Reconfigure (see below)
      regularizers_(),
      score_calculators_(),
      batches_(),
      models_(),
      processor_queue_(),
      cache_manager_(),
      score_manager_(),
      score_tracker_(),
      processors_() {
  Reconfigure(*rhs.config());

  std::vector<std::string> batch_name = rhs.batches_.keys();
  for (const auto& key : batch_name) {
    std::shared_ptr<Batch> value = rhs.batches_.get(key);
    if (value != nullptr) {
      batches_.set(key, value);  // store same batch as rhs (OK as batches here are read-only)
    }
  }

  std::vector<ModelName> model_name = rhs.models_.keys();
  for (const auto& key : model_name) {
    std::shared_ptr<const PhiMatrix> value = rhs.GetPhiMatrix(key);
    if (value != nullptr) {
      this->SetPhiMatrix(key, value->Duplicate());
    }
  }

  cache_manager_->CopyFrom(*rhs.cache_manager_);
  score_manager_->CopyFrom(*rhs.score_manager_);
  score_tracker_->CopyFrom(*rhs.score_tracker_);
}

Instance::~Instance() { }

std::shared_ptr<Instance> Instance::Duplicate() const {
  return std::shared_ptr<Instance>(new Instance(*this));
}

void Instance::RequestMasterComponentInfo(MasterComponentInfo* master_info) const {
  auto config = master_model_config_.get();
  if (config != nullptr) {
    master_info->mutable_config()->CopyFrom(*config);
  }

  for (const auto& key : regularizers_.keys()) {
    auto regularizer = regularizers_.get(key);
    if (regularizer == nullptr) {
      continue;
    }

    MasterComponentInfo::RegularizerInfo* info = master_info->add_regularizer();
    info->set_name(key);
    info->set_type(typeid(*regularizer).name());
  }

  for (const auto& key : score_calculators_.keys()) {
    auto score_calculator = score_calculators_.get(key);
    if (score_calculator == nullptr) {
      continue;
    }

    MasterComponentInfo::ScoreInfo* info = master_info->add_score();
    info->set_name(key);
    info->set_type(typeid(*score_calculator).name());
  }

  cache_manager_->RequestMasterComponentInfo(master_info);

  for (const auto& name : dictionaries()->keys()) {
    std::shared_ptr<Dictionary> dict = dictionaries()->get(name);
    if (dict == nullptr) {
      continue;
    }

    MasterComponentInfo::DictionaryInfo* info = master_info->add_dictionary();
    info->set_name(name);
    info->set_num_entries(dict->size());
    info->set_byte_size(dict->ByteSize());
  }

  for (const auto& name : batches_.keys()) {
    std::shared_ptr<Batch> batch = batches_.get(name);
    if (batch == nullptr) {
      continue;
    }

    MasterComponentInfo::BatchInfo* info = master_info->add_batch();
    info->set_name(name);
    info->set_num_tokens(batch->token_size());
    info->set_num_items(batch->item_size());
  }

  for (const auto& name : models_.keys()) {
    std::shared_ptr<const PhiMatrix> p_wt = this->GetPhiMatrix(name);
    if (p_wt != nullptr) {
      MasterComponentInfo::ModelInfo* info = master_info->add_model();
      info->set_name(p_wt->model_name());
      info->set_num_tokens(p_wt->token_size());
      info->set_num_topics(p_wt->topic_size());
      info->set_type(typeid(*p_wt).name());
      info->set_byte_size(p_wt->ByteSize());
    }
  }

  master_info->set_processor_queue_size(static_cast<int>(processor_queue_.size()));
  master_info->set_num_processors(static_cast<int>(processors_.size()));
}

CacheManager* Instance::cache_manager() {
  return cache_manager_.get();
}

ScoreManager* Instance::score_manager() {
  return score_manager_.get();
}

ScoreTracker* Instance::score_tracker() {
  return score_tracker_.get();
}

void Instance::DisposeModel(ModelName model_name) {
  models_.erase(model_name);
}

void Instance::CreateOrReconfigureRegularizer(const RegularizerConfig& config) {
  std::string regularizer_name = config.name();
  artm::RegularizerType regularizer_type = config.type();

  auto regularizer = regularizers_.get(regularizer_name);
  bool need_hot_reconfigure = (regularizer != nullptr);

  std::string config_blob;  // Used by CREATE_OR_RECONFIGURE_REGULARIZER marco
  if (config.has_config()) {
    config_blob = config.config();
  }

  // add here new case if adding new regularizer
  switch (regularizer_type) {
    case artm::RegularizerType_SmoothSparseTheta: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::SmoothSparseThetaConfig,
                                        ::artm::regularizer::SmoothSparseTheta);
      break;
    }

    case artm::RegularizerType_SmoothSparsePhi: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::SmoothSparsePhiConfig,
                                        ::artm::regularizer::SmoothSparsePhi);
      break;
    }

    case artm::RegularizerType_LabelRegularizationPhi: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::LabelRegularizationPhiConfig,
                                        ::artm::regularizer::LabelRegularizationPhi);
      break;
    }

    case artm::RegularizerType_DecorrelatorPhi: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::DecorrelatorPhiConfig,
                                        ::artm::regularizer::DecorrelatorPhi);
      break;
    }

    case artm::RegularizerType_MultiLanguagePhi: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::MultiLanguagePhiConfig,
                                        ::artm::regularizer::MultiLanguagePhi);
      break;
    }

    case artm::RegularizerType_SpecifiedSparsePhi: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::SpecifiedSparsePhiConfig,
                                        ::artm::regularizer::SpecifiedSparsePhi);
      break;
    }

    case artm::RegularizerType_ImproveCoherencePhi: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::ImproveCoherencePhiConfig,
                                        ::artm::regularizer::ImproveCoherencePhi);
      break;
    }

    case artm::RegularizerType_SmoothPtdw: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::SmoothPtdwConfig,
                                        ::artm::regularizer::SmoothPtdw);
      break;
    }

    case artm::RegularizerType_TopicSelectionTheta: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::TopicSelectionThetaConfig,
                                        ::artm::regularizer::TopicSelectionTheta);
      break;
    }

    case artm::RegularizerType_BitermsPhi: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::BitermsPhiConfig,
        ::artm::regularizer::BitermsPhi);
      break;
    }

    case artm::RegularizerType_HierarchySparsingTheta: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::HierarchySparsingThetaConfig,
        ::artm::regularizer::HierarchySparsingTheta);
      break;
    }

    case artm::RegularizerType_TopicSegmentationPtdw: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::TopicSegmentationPtdwConfig,
        ::artm::regularizer::TopicSegmentationPtdw);
      break;
    }

    case artm::RegularizerType_SmoothTimeInTopicsPhi: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::SmoothTimeInTopicsPhiConfig,
        ::artm::regularizer::SmoothTimeInTopicsPhi);
      break;
    }

    case artm::RegularizerType_NetPlsaPhi: {
      CREATE_OR_RECONFIGURE_REGULARIZER(::artm::NetPlsaPhiConfig,
        ::artm::regularizer::NetPlsaPhi);
      break;
    }

    default:
      BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
        "RegularizerConfig.type", regularizer_type));
  }

  this->regularizers()->set(regularizer_name, regularizer);
}

std::shared_ptr<ScoreCalculatorInterface> Instance::CreateScoreCalculator(const ScoreConfig& config) {
  std::string score_name = config.name();
  artm::ScoreType score_type = config.type();

  std::shared_ptr<artm::ScoreCalculatorInterface> score_calculator;

  // add here new case if adding new score
  switch (score_type) {
    case artm::ScoreType_Perplexity: {
      score_calculator.reset(new ::artm::score::Perplexity(config));
      break;
    }

    case artm::ScoreType_SparsityTheta: {
      score_calculator.reset(new ::artm::score::SparsityTheta(config));
      break;
    }

    case artm::ScoreType_SparsityPhi: {
      score_calculator.reset(new ::artm::score::SparsityPhi(config));
      break;
    }

    case artm::ScoreType_ItemsProcessed: {
      score_calculator.reset(new ::artm::score::ItemsProcessed(config));
      break;
    }

    case artm::ScoreType_TopTokens: {
      score_calculator.reset(new ::artm::score::TopTokens(config));
      break;
    }

    case artm::ScoreType_ThetaSnippet: {
      score_calculator.reset(new ::artm::score::ThetaSnippet(config));
      break;
    }

    case artm::ScoreType_TopicKernel: {
      score_calculator.reset(new ::artm::score::TopicKernel(config));
      break;
    }

    case artm::ScoreType_TopicMassPhi: {
      score_calculator.reset(new ::artm::score::TopicMassPhi(config));
      break;
    }

    case artm::ScoreType_ClassPrecision: {
      score_calculator.reset(new ::artm::score::ClassPrecision(config));
      break;
    }

    case artm::ScoreType_PeakMemory: {
      score_calculator.reset(new ::artm::score::PeakMemory(config));
      break;
    }

    case artm::ScoreType_BackgroundTokensRatio: {
      score_calculator.reset(new ::artm::score::BackgroundTokensRatio(config));
      break;
    }

    default:
      BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException("ScoreConfig.type", score_type));
  }

  score_calculator->set_instance(this);
  return score_calculator;
}

void Instance::DisposeRegularizer(const std::string& name) {
  regularizers_.erase(name);
}

void Instance::Reconfigure(const MasterModelConfig& master_config) {
  master_model_config_.set(std::make_shared<MasterModelConfig>(master_config));

  int target_processors_count = master_config.num_processors();
  if (!master_config.has_num_processors() || master_config.num_processors() < 0) {
    unsigned int n = std::thread::hardware_concurrency();
    if (n == 0) {
      LOG(INFO) << "MasterModelConfig.processors_count is set to 1 (default)";
      target_processors_count = 1;
    } else {
      LOG(INFO) << "MasterModelConfig.processors_count is automatically set to " << n;
      target_processors_count = n;
    }
  }

  score_calculators_.clear();
  for (int score_index = 0;
       score_index < master_config.score_config_size();
       ++score_index) {
    const ScoreConfig& score_config = master_config.score_config(score_index);
    auto score_calculator = CreateScoreCalculator(score_config);
    this->scores_calculators()->set(score_config.name(), score_calculator);
  }

  if (!is_configured_) {
    // First reconfiguration.
    cache_manager_.reset(new CacheManager(master_config.disk_cache_path(), this));
    score_manager_.reset(new ScoreManager(this));
    score_tracker_.reset(new ScoreTracker());

    is_configured_  = true;
  }

  {
    // Adjust size of processors_; cast size to int to avoid compiler warning.
    while (static_cast<int>(processors_.size()) > target_processors_count) {
      processors_.pop_back();
    }

    while (static_cast<int>(processors_.size()) < target_processors_count) {
      processors_.push_back(std::shared_ptr<Processor>(new Processor(this)));
    }
  }

  if (master_config.has_disk_cache_path()) {
    boost::filesystem::path dir(master_config.disk_cache_path());
    if (!boost::filesystem::is_directory(dir)) {
      if (!boost::filesystem::create_directory(dir)) {
        BOOST_THROW_EXCEPTION(DiskWriteException("Unable to create folder '" + master_config.disk_cache_path() + "'"));
      }
    }
  }
}

std::shared_ptr<const ::artm::core::PhiMatrix>
Instance::GetPhiMatrix(ModelName model_name) const {
  return models_.get(model_name);
}

std::shared_ptr<const ::artm::core::PhiMatrix>
Instance::GetPhiMatrixSafe(ModelName model_name) const {
  std::shared_ptr<const PhiMatrix> retval = models_.get(model_name);
  if (retval == nullptr) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Model " + model_name + " does not exist"));
  }
  return retval;
}

void Instance::SetPhiMatrix(ModelName model_name, std::shared_ptr< ::artm::core::PhiMatrix> phi_matrix) {
  models_.erase(model_name);
  if (phi_matrix != nullptr) {
    models_.set(model_name, phi_matrix);
  }
}

}  // namespace core
}  // namespace artm
