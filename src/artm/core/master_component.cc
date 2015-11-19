// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/master_component.h"

#include <algorithm>
#include <fstream>  // NOLINT
#include <vector>
#include <set>
#include <sstream>

#include "boost/uuid/uuid_generators.hpp"
#include "boost/thread.hpp"

#include "glog/logging.h"

#include "artm/regularizer_interface.h"
#include "artm/score_calculator_interface.h"

#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/data_loader.h"
#include "artm/core/batch_manager.h"
#include "artm/core/cache_manager.h"
#include "artm/core/instance.h"
#include "artm/core/processor.h"
#include "artm/core/phi_matrix_operations.h"
#include "artm/core/topic_model.h"
#include "artm/core/scores_merger.h"
#include "artm/core/merger.h"
#include "artm/core/dense_phi_matrix.h"
#include "artm/core/template_manager.h"

namespace artm {
namespace core {

MasterComponent::MasterComponent(const MasterComponentConfig& config)
    : instance_(std::make_shared<Instance>(config)) {
}

MasterComponent::MasterComponent(const MasterComponent& rhs)
    : instance_(rhs.instance_->Duplicate()) {
}

MasterComponent::~MasterComponent() {}

std::shared_ptr<MasterComponent> MasterComponent::Duplicate() const {
  return std::shared_ptr<MasterComponent>(new MasterComponent(*this));
}

void MasterComponent::CreateOrReconfigureModel(const ModelConfig& config) {
  if ((config.class_weight_size() != 0 || config.class_id_size() != 0) && !config.use_sparse_bow()) {
    std::stringstream ss;
    ss << "You have configured use_sparse_bow=false. "
       << "Fields ModelConfig.class_id and ModelConfig.class_weight not supported in this mode.";
    BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
  }

  LOG(INFO) << "MasterComponent::CreateOrReconfigureModel() with " << Helpers::Describe(config);
  instance_->CreateOrReconfigureModel(config);
}

void MasterComponent::DisposeModel(ModelName model_name) {
  instance_->DisposeModel(model_name);
}

void MasterComponent::CreateOrReconfigureRegularizer(const RegularizerConfig& config) {
  instance_->CreateOrReconfigureRegularizer(config);
}

void MasterComponent::DisposeRegularizer(const std::string& name) {
  instance_->DisposeRegularizer(name);
}

void MasterComponent::CreateOrReconfigureDictionary(const DictionaryData& data) {
  instance_->CreateOrReconfigureDictionary(data);
}

void MasterComponent::DisposeDictionary(const std::string& name) {
  instance_->DisposeDictionary(name);
}

void MasterComponent::ExportDictionary(const ExportDictionaryArgs& args) {
  if (boost::filesystem::exists(args.file_name()))
    BOOST_THROW_EXCEPTION(DiskWriteException("File already exists: " + args.file_name()));

  std::ofstream fout(args.file_name(), std::ofstream::binary);
  if (!fout.is_open())
    BOOST_THROW_EXCEPTION(DiskReadException("Unable to create file " + args.file_name()));

  std::shared_ptr<Dictionary> dict_ptr = instance_->dictionary(args.dictionary_name());
  if (dict_ptr == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation("Dictionary " +
        args.dictionary_name() + " does not exist or has no tokens"));

  LOG(INFO) << "Exporting dictionary " << args.dictionary_name() << " to " << args.file_name();

  const int token_size = dict_ptr->size();
  
  // ToDo: MelLain
  // Add ability to save and load several token_dict_data
  // int tokens_per_chunk = std::min<int>(token_size, 3e+7);

  DictionaryData token_dict_data;
  token_dict_data.set_name(args.dictionary_name());
  DictionaryData cooc_dict_data;
  int current_cooc_length = 0;
  const int max_cooc_length = 1e+7;

  const char version = 0;
  fout << version;
  for (int token_id = 0; token_id < token_size; ++token_id) {
    auto entry = dict_ptr->entry(token_id);
    token_dict_data.add_token(entry->token().keyword);
    token_dict_data.add_class_id(entry->token().class_id);
    token_dict_data.add_token_value(entry->token_value());
    token_dict_data.add_token_tf(entry->token_tf());
    token_dict_data.add_token_df(entry->token_df());

    auto cooc_info = dict_ptr->cooc_info(entry->token());

    for (auto& iter = cooc_info->begin(); iter != cooc_info->end(); ++iter) {
      cooc_dict_data.add_cooc_first_index(token_id);
      cooc_dict_data.add_cooc_second_index(iter->first);
      cooc_dict_data.add_cooc_value(iter->second);
      current_cooc_length++;
    }

    if (current_cooc_length >= max_cooc_length) {
      std::string str = cooc_dict_data.SerializeAsString();
      fout << str.size();
      fout << str;
      cooc_dict_data.clear_cooc_first_index();
      cooc_dict_data.clear_cooc_second_index();
      cooc_dict_data.clear_cooc_value();
      current_cooc_length = 0;
    }

    if ((token_id + 1) == token_size) {
      std::string str = cooc_dict_data.SerializeAsString();
      fout << str.size();
      fout << str;

      std::string str = token_dict_data.SerializeAsString();
      fout << str.size();
      fout << str;
    }
  }

  fout.close();
  LOG(INFO) << "Export completed, token_size = " << dict_ptr->size();
}

void MasterComponent::ImportDictionary(const ImportDictionaryArgs& args) {
  std::ifstream fin(args.file_name(), std::ifstream::binary);
  if (!fin.is_open())
    BOOST_THROW_EXCEPTION(DiskReadException("Unable to open file " + args.file_name()));

  LOG(INFO) << "Importing dictionary " << args.dictionary_name() << " from " << args.file_name();

  char version;
  fin >> version;
  if (version != 0) {
    std::stringstream ss;
    ss << "Unsupported fromat version: " << static_cast<int>(version);
    BOOST_THROW_EXCEPTION(DiskReadException(ss.str()));
  }

  std::vector<std::shared_ptr<::artm::DictionaryData> > temp_data;
  while (!fin.eof()) {
    int length;
    fin >> length;
    if (fin.eof())
      break;

    if (length <= 0)
      BOOST_THROW_EXCEPTION(CorruptedMessageException("Unable to read from " + args.file_name()));

    std::string buffer(length, '\0');
    fin.read(&buffer[0], length);
    ::artm::DictionaryData dict_data;
    if (!dict_data.ParseFromArray(buffer.c_str(), length))
      BOOST_THROW_EXCEPTION(CorruptedMessageException("Unable to read from " + args.file_name()));

    temp_data.push_back(std::make_shared<::artm::DictionaryData>(dict_data));
  }

  fin.close();

  // now last element of temp_data contains ptr to tokens part of dictionary
  std::string dict_name = temp_data.back()->name();

  int token_size = temp_data.back()->token_size();
  if (token_size <= 0)
    BOOST_THROW_EXCEPTION(CorruptedMessageException("Unable to read from " + args.file_name()));

  instance_->CreateOrReconfigureDictionary(*(temp_data.back().get()));
  temp_data.pop_back();

  int temp_size = temp_data.size();
  for (int i = 0; i < temp_size; ++i) {
    instance_->dictionary(dict_name)->Append(*(temp_data.back().get()));
    temp_data.pop_back();
  }

  LOG(INFO) << "Import completed, token_size = " << token_size;
}

void MasterComponent::ImportBatches(const ImportBatchesArgs& args) {
  if (args.batch_name_size() != args.batch_size())
    BOOST_THROW_EXCEPTION(InvalidOperation("ImportBatchesArgs: batch_name_size() != batch_size()"));

  for (int i = 0; i < args.batch_name_size(); ++i) {
    std::shared_ptr<Batch> batch = std::make_shared<Batch>(args.batch(i));
    Helpers::FixAndValidate(batch.get(), /* throw_error =*/ true);
    instance_->batches()->set(args.batch_name(i), batch);
  }
}

void MasterComponent::DisposeBatches(const DisposeBatchesArgs& args) {
  for (auto& batch_name : args.batch_name())
    instance_->batches()->erase(batch_name);
}


void MasterComponent::SynchronizeModel(const SynchronizeModelArgs& args) {
  instance_->merger()->ForceSynchronizeModel(args);
}

void MasterComponent::ExportModel(const ExportModelArgs& args) {
  if (boost::filesystem::exists(args.file_name()))
    BOOST_THROW_EXCEPTION(DiskWriteException("File already exists: " + args.file_name()));

  std::ofstream fout(args.file_name(), std::ofstream::binary);
  if (!fout.is_open())
    BOOST_THROW_EXCEPTION(DiskReadException("Unable to create file " + args.file_name()));

  std::shared_ptr<const TopicModel> topic_model = instance_->merger()->GetLatestTopicModel(args.model_name());
  std::shared_ptr<const PhiMatrix> phi_matrix = instance_->merger()->GetPhiMatrix(args.model_name());
  if (topic_model == nullptr && phi_matrix == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation("Model " + args.model_name() + " does not exist"));
  const PhiMatrix& n_wt = (topic_model != nullptr) ? topic_model->GetPwt() : *phi_matrix;

  LOG(INFO) << "Exporting model " << args.model_name() << " to " << args.file_name();

  const int token_size = n_wt.token_size();
  if (token_size == 0)
    BOOST_THROW_EXCEPTION(InvalidOperation("Model " + args.model_name() + " has no tokens, export failed"));

  int tokens_per_chunk = std::min<int>(token_size, 100 * 1024 * 1024 / n_wt.topic_size());

  ::artm::GetTopicModelArgs get_topic_model_args;
  get_topic_model_args.set_model_name(args.model_name());
  get_topic_model_args.set_request_type(::artm::GetTopicModelArgs_RequestType_Nwt);
  get_topic_model_args.set_matrix_layout(::artm::GetTopicModelArgs_MatrixLayout_Sparse);
  get_topic_model_args.mutable_token()->Reserve(tokens_per_chunk);
  get_topic_model_args.mutable_class_id()->Reserve(tokens_per_chunk);

  const char version = 0;
  fout << version;
  for (int token_id = 0; token_id < token_size; ++token_id) {
    Token token = n_wt.token(token_id);
    get_topic_model_args.add_token(token.keyword);
    get_topic_model_args.add_class_id(token.class_id);

    if (((token_id + 1) == token_size) || (get_topic_model_args.token_size() >= tokens_per_chunk)) {
      ::artm::TopicModel external_topic_model;
      PhiMatrixOperations::RetrieveExternalTopicModel(n_wt, get_topic_model_args, &external_topic_model);
      std::string str = external_topic_model.SerializeAsString();
      fout << str.size();
      fout << str;
      get_topic_model_args.clear_class_id();
      get_topic_model_args.clear_token();
    }
  }

  fout.close();
  LOG(INFO) << "Export completed, token_size = " << n_wt.token_size()
            << ", topic_size = " << n_wt.topic_size();
}

void MasterComponent::ImportModel(const ImportModelArgs& args) {
  std::ifstream fin(args.file_name(), std::ifstream::binary);
  if (!fin.is_open())
    BOOST_THROW_EXCEPTION(DiskReadException("Unable to open file " + args.file_name()));

  LOG(INFO) << "Importing model " << args.model_name() << " from " << args.file_name();

  char version;
  fin >> version;
  if (version != 0) {
    std::stringstream ss;
    ss << "Unsupported fromat version: " << static_cast<int>(version);
    BOOST_THROW_EXCEPTION(DiskReadException(ss.str()));
  }

  std::shared_ptr<DensePhiMatrix> target;
  while (!fin.eof()) {
    int length;
    fin >> length;
    if (fin.eof())
      break;

    if (length <= 0)
      BOOST_THROW_EXCEPTION(CorruptedMessageException("Unable to read from " + args.file_name()));

    std::string buffer(length, '\0');
    fin.read(&buffer[0], length);
    ::artm::TopicModel topic_model;
    if (!topic_model.ParseFromArray(buffer.c_str(), length))
      BOOST_THROW_EXCEPTION(CorruptedMessageException("Unable to read from " + args.file_name()));

    topic_model.set_name(args.model_name());

    if (target == nullptr)
      target = std::make_shared<DensePhiMatrix>(args.model_name(), topic_model.topic_name());

    PhiMatrixOperations::ApplyTopicModelOperation(topic_model, 1.0f, target.get());
  }

  fin.close();

  if (target == nullptr)
    BOOST_THROW_EXCEPTION(CorruptedMessageException("Unable to read from " + args.file_name()));

  instance_->merger()->SetPhiMatrix(args.model_name(), target);
  LOG(INFO) << "Import completed, token_size = " << target->token_size()
    << ", topic_size = " << target->topic_size();
}

void MasterComponent::AttachModel(const AttachModelArgs& args, int address_length, float* address) {
  ModelName model_name = args.model_name();
  LOG(INFO) << "Attaching model " << model_name << " to " << address << " (" << address_length << " bytes)";

  std::shared_ptr<const PhiMatrix> phi_matrix = instance_->merger()->GetPhiMatrix(model_name);
  if (phi_matrix == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation("Model " + model_name + " does not exist"));

  PhiMatrixFrame* frame = dynamic_cast<PhiMatrixFrame*>(const_cast<PhiMatrix*>(phi_matrix.get()));
  if (frame == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation("Unable to attach to model " + model_name));

  std::shared_ptr<AttachedPhiMatrix> attached = std::make_shared<AttachedPhiMatrix>(address_length, address, frame);
  instance_->merger()->SetPhiMatrix(model_name, attached);
}

void MasterComponent::InitializeModel(const InitializeModelArgs& args) {
  LOG(INFO) << "MasterComponent::InitializeModel() with " << Helpers::Describe(args);
  instance_->merger()->InitializeModel(args);
}

void MasterComponent::Reconfigure(const MasterComponentConfig& config) {
  LOG(INFO) << "MasterComponent::Reconfigure() with " << Helpers::Describe(config);

  if (instance_->schema()->config().disk_path() != config.disk_path())
    BOOST_THROW_EXCEPTION(InvalidOperation("Changing disk_path is not supported."));

  instance_->Reconfigure(config);
}

bool MasterComponent::RequestTopicModel(const ::artm::GetTopicModelArgs& get_model_args,
                                        ::artm::TopicModel* topic_model) {
  return instance_->merger()->RetrieveExternalTopicModel(get_model_args, topic_model);
}

void MasterComponent::RequestRegularizerState(RegularizerName regularizer_name,
                                              ::artm::RegularizerInternalState* regularizer_state) {
  instance_->merger()->RequestRegularizerState(regularizer_name, regularizer_state);
}

bool MasterComponent::RequestScore(const GetScoreValueArgs& get_score_args,
                                   ScoreData* score_data) {
  if (!get_score_args.has_batch()) {
    return instance_->merger()->RequestScore(get_score_args, score_data);
  }

  if (instance_->processor_size() == 0)
    BOOST_THROW_EXCEPTION(InternalError("No processors exist in the master component"));
  instance_->processor(0)->FindThetaMatrix(
    get_score_args.batch(), GetThetaMatrixArgs(), nullptr, get_score_args, score_data);
  return true;
}

void MasterComponent::RequestDictionary(DictionaryName regularizer_name,
                                        ::artm::DictionaryData* dictionary_data) {
  instance_->RequestDictionary(dictionary_name, dictionary_data);
}

void MasterComponent::RequestMasterComponentInfo(MasterComponentInfo* master_info) const {
  std::shared_ptr<InstanceSchema> instance_schema = instance_->schema();
  this->instance_->RequestMasterComponentInfo(master_info);
}

void MasterComponent::RequestProcessBatches(const ProcessBatchesArgs& process_batches_args,
                                            ProcessBatchesResult* process_batches_result) {
  BatchManager batch_manager;
  RequestProcessBatchesImpl(process_batches_args, &batch_manager, /* async =*/ false, process_batches_result);
}

void MasterComponent::AsyncRequestProcessBatches(const ProcessBatchesArgs& process_batches_args,
                                                 BatchManager *batch_manager) {
  RequestProcessBatchesImpl(process_batches_args, batch_manager, /* async =*/ true, nullptr);
}

void MasterComponent::RequestProcessBatchesImpl(const ProcessBatchesArgs& process_batches_args,
                                                BatchManager* batch_manager, bool async,
                                                ProcessBatchesResult* process_batches_result) {
  LOG(INFO) << "MasterComponent::RequestProcessBatches() with " << Helpers::Describe(process_batches_args);
  std::shared_ptr<InstanceSchema> schema = instance_->schema();
  const MasterComponentConfig& config = schema->config();

  const ProcessBatchesArgs& args = process_batches_args;  // short notation
  ModelName model_name = args.pwt_source_name();
  ModelConfig model_config;
  model_config.set_name(model_name);
  if (args.has_inner_iterations_count()) model_config.set_inner_iterations_count(args.inner_iterations_count());
  if (args.has_stream_name()) model_config.set_stream_name(args.stream_name());
  model_config.mutable_regularizer_name()->CopyFrom(args.regularizer_name());
  model_config.mutable_regularizer_tau()->CopyFrom(args.regularizer_tau());
  model_config.mutable_class_id()->CopyFrom(args.class_id());
  model_config.mutable_class_weight()->CopyFrom(args.class_weight());
  if (args.has_reuse_theta()) model_config.set_reuse_theta(args.reuse_theta());
  if (args.has_opt_for_avx()) model_config.set_opt_for_avx(args.opt_for_avx());
  if (args.has_use_sparse_bow()) model_config.set_use_sparse_bow(args.use_sparse_bow());
  if (args.has_model_name_cache()) model_config.set_model_name_cache(args.model_name_cache());
  if (args.has_predict_class_id()) model_config.set_predict_class_id(args.predict_class_id());

  std::shared_ptr<const TopicModel> topic_model = instance_->merger()->GetLatestTopicModel(model_name);
  std::shared_ptr<const PhiMatrix> phi_matrix = instance_->merger()->GetPhiMatrix(model_name);
  if (topic_model == nullptr && phi_matrix == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation("Model " + model_name + " does not exist"));
  const PhiMatrix& p_wt = (topic_model != nullptr) ? topic_model->GetPwt() : *phi_matrix;

  if (args.has_nwt_target_name()) {
    if (args.nwt_target_name() == args.pwt_source_name())
      BOOST_THROW_EXCEPTION(InvalidOperation(
        "ProcessBatchesArgs.pwt_source_name == ProcessBatchesArgs.nwt_target_name"));

    auto nwt_target(std::make_shared<DensePhiMatrix>(args.nwt_target_name(), p_wt.topic_name()));
    nwt_target->Reshape(p_wt);
    instance_->merger()->SetPhiMatrix(args.nwt_target_name(), nwt_target);
  }

  model_config.set_topics_count(p_wt.topic_size());
  model_config.mutable_topic_name()->CopyFrom(p_wt.topic_name());
  Helpers::FixAndValidate(&model_config, /* throw_error =*/ true);

  if (async && args.theta_matrix_type() != ProcessBatchesArgs_ThetaMatrixType_None)
    BOOST_THROW_EXCEPTION(InvalidOperation(
    "ArtmAsyncProcessBatches require ProcessBatchesArgs.theta_matrix_type to be set to None"));

  // The code below must not use cache_manger in async mode.
  // Since cache_manager lives on stack it will be destroyed once we return from this function.
  // Therefore, no pointers to cache_manager should exist upon return from RequestProcessBatchesImpl.
  CacheManager cache_manager;
  ScoresMerger* scores_merger = instance_->merger()->scores_merger();

  bool return_theta = false;
  bool return_ptdw = false;
  CacheManager* ptdw_cache_manager_ptr = nullptr;
  CacheManager* theta_cache_manager_ptr = nullptr;
  switch (args.theta_matrix_type()) {
    case ProcessBatchesArgs_ThetaMatrixType_Cache:
      if (instance_->schema()->config().cache_theta())
        theta_cache_manager_ptr = instance_->cache_manager();
      break;
    case ProcessBatchesArgs_ThetaMatrixType_Dense:
    case ProcessBatchesArgs_ThetaMatrixType_Sparse:
      theta_cache_manager_ptr = &cache_manager;
      return_theta = true;
      break;
    case ProcessBatchesArgs_ThetaMatrixType_DensePtdw:
    case ProcessBatchesArgs_ThetaMatrixType_SparsePtdw:
      ptdw_cache_manager_ptr = &cache_manager;
      return_ptdw = true;
  }

  if (args.reset_scores())
    scores_merger->ResetScores(model_name);

  if (args.batch_filename_size() < config.processors_count()) {
    LOG_FIRST_N(INFO, 1) << "Batches count (=" << args.batch_filename_size()
                         << ") is smaller than processors threads count (="
                         << config.processors_count()
                         << "), which may cause suboptimal performance.";
  }

  for (int batch_index = 0; batch_index < args.batch_filename_size(); ++batch_index) {
    boost::uuids::uuid task_id = boost::uuids::random_generator()();
    batch_manager->Add(task_id, std::string(), model_name);

    auto pi = std::make_shared<ProcessorInput>();
    pi->set_notifiable(batch_manager);
    pi->set_scores_merger(scores_merger);
    pi->set_cache_manager(theta_cache_manager_ptr);
    pi->set_ptdw_cache_manager(ptdw_cache_manager_ptr);
    pi->set_model_name(model_name);
    pi->set_batch_filename(args.batch_filename(batch_index));
    pi->set_batch_weight(args.batch_weight(batch_index));
    pi->mutable_model_config()->CopyFrom(model_config);
    pi->set_task_id(task_id);
    pi->set_caller(ProcessorInput::Caller::ProcessBatches);

    if (args.reuse_theta())
      pi->set_reuse_theta_cache_manager(instance_->cache_manager());

    if (args.has_nwt_target_name())
      pi->set_nwt_target_name(args.nwt_target_name());

    instance_->processor_queue()->push(pi);
  }

  if (async)
    return;

  while (!batch_manager->IsEverythingProcessed()) {
    boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));
  }

  for (int score_index = 0; score_index < config.score_config_size(); ++score_index) {
    ScoreName score_name = config.score_config(score_index).name();
    ScoreData score_data;
    if (scores_merger->RequestScore(schema, model_name, score_name, &score_data))
      process_batches_result->add_score_data()->Swap(&score_data);
  }

  GetThetaMatrixArgs get_theta_matrix_args;
  get_theta_matrix_args.set_model_name(model_name);
  switch (args.theta_matrix_type()) {
    case ProcessBatchesArgs_ThetaMatrixType_Dense:
    case ProcessBatchesArgs_ThetaMatrixType_DensePtdw:
      get_theta_matrix_args.set_matrix_layout(GetThetaMatrixArgs_MatrixLayout_Dense);
      break;
    case ProcessBatchesArgs_ThetaMatrixType_Sparse:
    case ProcessBatchesArgs_ThetaMatrixType_SparsePtdw:
      get_theta_matrix_args.set_matrix_layout(GetThetaMatrixArgs_MatrixLayout_Sparse);
      break;
  }

  if (args.has_theta_matrix_type())
    cache_manager.RequestThetaMatrix(get_theta_matrix_args, process_batches_result->mutable_theta_matrix());
}

void MasterComponent::MergeModel(const MergeModelArgs& merge_model_args) {
  LOG(INFO) << "MasterComponent::MergeModel() with " << Helpers::Describe(merge_model_args);
  if (merge_model_args.nwt_source_name_size() == 0)
    BOOST_THROW_EXCEPTION(InvalidOperation("MergeModelArgs.nwt_source_name must not be empty"));
  if (merge_model_args.nwt_source_name_size() != merge_model_args.source_weight_size())
    BOOST_THROW_EXCEPTION(InvalidOperation(
      "MergeModelArgs.nwt_source_name_size() != MergeModelArgs.source_weight_size()"));

  std::shared_ptr<DensePhiMatrix> nwt_target;
  std::stringstream ss;
  for (int i = 0; i < merge_model_args.nwt_source_name_size(); ++i) {
    ModelName model_name = merge_model_args.nwt_source_name(i);
    ss << (i == 0 ? "" : ", ") << model_name;

    float weight = merge_model_args.source_weight(i);

    std::shared_ptr<const TopicModel> topic_model = instance_->merger()->GetLatestTopicModel(model_name);
    std::shared_ptr<const PhiMatrix> phi_matrix = instance_->merger()->GetPhiMatrix(model_name);
    if (topic_model == nullptr && phi_matrix == nullptr) {
      LOG(WARNING) << "Model " << model_name << " does not exist";
      continue;
    }
    const PhiMatrix& n_wt = (topic_model != nullptr) ? topic_model->GetNwt() : *phi_matrix;

    if (nwt_target == nullptr) {
      nwt_target = std::make_shared<DensePhiMatrix>(
        merge_model_args.nwt_target_name(),
        merge_model_args.topic_name_size() != 0 ? merge_model_args.topic_name() : n_wt.topic_name());
    }

    if (n_wt.token_size() > 0) {
      ::artm::TopicModel topic_model;
      PhiMatrixOperations::RetrieveExternalTopicModel(n_wt, GetTopicModelArgs(), &topic_model);
      PhiMatrixOperations::ApplyTopicModelOperation(topic_model, weight, nwt_target.get());
    }
  }

  if (nwt_target == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation(
      "ArtmMergeModel() have not found any models to merge. "
      "Verify that at least one of the following models exist: " + ss.str()));
  instance_->merger()->SetPhiMatrix(merge_model_args.nwt_target_name(), nwt_target);
}

void MasterComponent::RegularizeModel(const RegularizeModelArgs& regularize_model_args) {
  LOG(INFO) << "MasterComponent::RegularizeModel() with " << Helpers::Describe(regularize_model_args);
  const std::string& pwt_source_name = regularize_model_args.pwt_source_name();
  const std::string& nwt_source_name = regularize_model_args.nwt_source_name();
  const std::string& rwt_target_name = regularize_model_args.rwt_target_name();

  if (!regularize_model_args.has_pwt_source_name())
    BOOST_THROW_EXCEPTION(InvalidOperation("RegularizeModelArgs.pwt_source_name is missing"));
  if (!regularize_model_args.has_nwt_source_name())
    BOOST_THROW_EXCEPTION(InvalidOperation("RegularizeModelArgs.nwt_source_name is missing"));
  if (!regularize_model_args.has_rwt_target_name())
    BOOST_THROW_EXCEPTION(InvalidOperation("RegularizeModelArgs.rwt_target_name is missing"));

  std::shared_ptr<const PhiMatrix> nwt_phi_matrix = instance_->merger()->GetPhiMatrix(nwt_source_name);
  std::shared_ptr<const TopicModel> nwt_topic_model = instance_->merger()->GetLatestTopicModel(nwt_source_name);
  if (nwt_phi_matrix == nullptr && nwt_topic_model == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation("Model " + nwt_source_name + " does not exist"));
  const PhiMatrix& n_wt = (nwt_topic_model != nullptr) ? nwt_topic_model->GetNwt() : *nwt_phi_matrix;

  std::shared_ptr<const PhiMatrix> pwt_phi_matrix = instance_->merger()->GetPhiMatrix(pwt_source_name);
  std::shared_ptr<const TopicModel> pwt_topic_model = instance_->merger()->GetLatestTopicModel(pwt_source_name);
  if (pwt_phi_matrix == nullptr && pwt_topic_model == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation("Model " + pwt_source_name + " does not exist"));
  const PhiMatrix& p_wt = (pwt_topic_model != nullptr) ? pwt_topic_model->GetPwt() : *pwt_phi_matrix;

  auto rwt_target(std::make_shared<DensePhiMatrix>(rwt_target_name, nwt_phi_matrix->topic_name()));
  rwt_target->Reshape(*nwt_phi_matrix);
  PhiMatrixOperations::InvokePhiRegularizers(instance_->schema(), regularize_model_args.regularizer_settings(),
                                             p_wt, n_wt, rwt_target.get());
  instance_->merger()->SetPhiMatrix(rwt_target_name, rwt_target);
}

void MasterComponent::NormalizeModel(const NormalizeModelArgs& normalize_model_args) {
  LOG(INFO) << "MasterComponent::NormalizeModel() with " << Helpers::Describe(normalize_model_args);
  const std::string& pwt_target_name = normalize_model_args.pwt_target_name();
  const std::string& nwt_source_name = normalize_model_args.nwt_source_name();
  const std::string& rwt_source_name = normalize_model_args.rwt_source_name();

  if (!normalize_model_args.has_pwt_target_name())
    BOOST_THROW_EXCEPTION(InvalidOperation("NormalizeModelArgs.pwt_target_name is missing"));
  if (!normalize_model_args.has_nwt_source_name())
    BOOST_THROW_EXCEPTION(InvalidOperation("NormalizeModelArgs.pwt_target_name is missing"));

  std::shared_ptr<const TopicModel> nwt_topic_model = instance_->merger()->GetLatestTopicModel(nwt_source_name);
  std::shared_ptr<const PhiMatrix> nwt_phi_matrix = instance_->merger()->GetPhiMatrix(nwt_source_name);
  if ((nwt_topic_model == nullptr) && (nwt_phi_matrix == nullptr))
    BOOST_THROW_EXCEPTION(InvalidOperation("Model " + nwt_source_name + " does not exist"));
  const PhiMatrix& n_wt = (nwt_topic_model != nullptr) ? nwt_topic_model->GetPwt() : *nwt_phi_matrix;

  const PhiMatrix* r_wt = nullptr;
  std::shared_ptr<const TopicModel> rwt_topic_model = instance_->merger()->GetLatestTopicModel(rwt_source_name);
  std::shared_ptr<const PhiMatrix> rwt_phi_matrix = instance_->merger()->GetPhiMatrix(rwt_source_name);
  if (normalize_model_args.has_rwt_source_name()) {
    if ((rwt_topic_model == nullptr) && (rwt_phi_matrix == nullptr))
      BOOST_THROW_EXCEPTION(InvalidOperation("Model " + rwt_source_name + " does not exist"));
    r_wt = (rwt_topic_model != nullptr) ? &rwt_topic_model->GetPwt() : rwt_phi_matrix.get();
  }

  auto pwt_target(std::make_shared<DensePhiMatrix>(pwt_target_name, n_wt.topic_name()));
  pwt_target->Reshape(n_wt);
  if (r_wt == nullptr) PhiMatrixOperations::FindPwt(n_wt, pwt_target.get());
  else                 PhiMatrixOperations::FindPwt(n_wt, *r_wt, pwt_target.get());
  instance_->merger()->SetPhiMatrix(pwt_target_name, pwt_target);
}

void MasterComponent::OverwriteTopicModel(const ::artm::TopicModel& topic_model) {
  instance_->merger()->OverwriteTopicModel(topic_model);
}

bool MasterComponent::RequestThetaMatrix(const GetThetaMatrixArgs& get_theta_args,
                                         ::artm::ThetaMatrix* theta_matrix) {
  if (!get_theta_args.has_batch()) {
    return instance_->cache_manager()->RequestThetaMatrix(get_theta_args, theta_matrix);
  } else {
    if (instance_->processor_size() == 0)
      BOOST_THROW_EXCEPTION(InternalError("No processors exist in the master component"));
    instance_->processor(0)->FindThetaMatrix(
      get_theta_args.batch(), get_theta_args, theta_matrix, GetScoreValueArgs(), nullptr);
    return true;
  }
}

bool MasterComponent::WaitIdle(const WaitIdleArgs& args) {
  int timeout = args.timeout_milliseconds();
  LOG_IF(WARNING, timeout == 0) << "WaitIdleArgs.timeout_milliseconds == 0";
  WaitIdleArgs new_args;
  new_args.CopyFrom(args);
  auto time_start = boost::posix_time::microsec_clock::local_time();

  bool retval = instance_->data_loader()->WaitIdle(args);
  if (!retval) return false;

  auto time_end = boost::posix_time::microsec_clock::local_time();
  if (timeout != -1) {
    timeout -= (time_end - time_start).total_milliseconds();
    new_args.set_timeout_milliseconds(timeout);
  }

  return instance_->merger()->WaitIdle(new_args);
}

void MasterComponent::InvokeIteration(const InvokeIterationArgs& args) {
  if (args.reset_scores())
    instance_->merger()->ForceResetScores(ModelName());

  instance_->data_loader()->InvokeIteration(args);
}

bool MasterComponent::AddBatch(const AddBatchArgs& args) {
  int timeout = args.timeout_milliseconds();
  LOG_IF(WARNING, timeout == 0) << "AddBatchArgs.timeout_milliseconds == 0";
  if (args.reset_scores())
    instance_->merger()->ForceResetScores(ModelName());

  return instance_->data_loader()->AddBatch(args);
}

}  // namespace core
}  // namespace artm
