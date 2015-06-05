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
#include "artm/core/merger.h"

namespace artm {
namespace core {

MasterComponent::MasterComponent(int id, const MasterComponentConfig& config)
    : is_configured_(false),
      master_id_(id),
      config_(std::make_shared<MasterComponentConfig>(config)),
      instance_(nullptr) {
  LOG(INFO) << "Creating MasterComponent (id=" << master_id_ << ")...";
  Reconfigure(config);
}

MasterComponent::~MasterComponent() {
  LOG(INFO) << "Disposing MasterComponent (id=" << master_id_ << ")...";
}

int MasterComponent::id() const {
  return master_id_;
}

void MasterComponent::CreateOrReconfigureModel(const ModelConfig& config) {
  if ((config.class_weight_size() != 0 || config.class_id_size() != 0) && !config.use_sparse_bow()) {
    std::stringstream ss;
    ss << "You have configured use_sparse_bow=false. "
       << "Fields ModelConfig.class_id and ModelConfig.class_weight not supported in this mode.";
    BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
  }

  LOG(INFO) << "Merger::CreateOrReconfigureModel() with " << Helpers::Describe(config);
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

void MasterComponent::CreateOrReconfigureDictionary(const DictionaryConfig& config) {
  instance_->CreateOrReconfigureDictionary(config);
}

void MasterComponent::DisposeDictionary(const std::string& name) {
  instance_->DisposeDictionary(name);
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

  std::shared_ptr<const ::artm::core::TopicModel> topic_model =
    instance_->merger()->GetLatestTopicModel(args.model_name());
  if (topic_model == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation("Model " + args.model_name() + " does not exist"));

  LOG(INFO) << "Exporting model " << args.model_name() << " to " << args.file_name();

  const int token_size = topic_model->token_size();
  int tokens_per_chunk = std::min<int>(token_size, 100 * 1024 * 1024 / topic_model->topic_size());

  ::artm::GetTopicModelArgs get_topic_model_args;
  get_topic_model_args.set_model_name(args.model_name());
  get_topic_model_args.set_request_type(::artm::GetTopicModelArgs_RequestType_Nwt);
  get_topic_model_args.set_use_sparse_format(true);
  get_topic_model_args.mutable_token()->Reserve(tokens_per_chunk);
  get_topic_model_args.mutable_class_id()->Reserve(tokens_per_chunk);
  for (int token_id = 0; token_id < token_size; ++token_id) {
    Token token = topic_model->token(token_id);
    get_topic_model_args.add_token(token.keyword);
    get_topic_model_args.add_class_id(token.class_id);

    if (((token_id + 1) == token_size) || (get_topic_model_args.token_size() >= tokens_per_chunk)) {
      ::artm::TopicModel external_topic_model;
      topic_model->RetrieveExternalTopicModel(get_topic_model_args, &external_topic_model);
      std::string str = external_topic_model.SerializeAsString();
      fout << str.size();
      fout << str;
      get_topic_model_args.clear_class_id();
      get_topic_model_args.clear_token();
    }
  }

  fout.close();
  LOG(INFO) << "Export completed, token_size = " << topic_model->token_size()
            << ", topic_size = " << topic_model->topic_size();
}

void MasterComponent::ImportModel(const ImportModelArgs& args) {
  std::ifstream fin(args.file_name(), std::ifstream::binary);
  if (!fin.is_open())
    BOOST_THROW_EXCEPTION(DiskReadException("Unable to open file " + args.file_name()));

  LOG(INFO) << "Importing model " << args.model_name() << " from " << args.file_name();

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
    OverwriteTopicModel(topic_model);
  }

  fin.close();
  WaitIdle(WaitIdleArgs());

  SynchronizeModelArgs sync_args;
  sync_args.set_model_name(args.model_name());
  sync_args.set_apply_weight(1.0f);
  sync_args.set_decay_weight(0.0f);
  sync_args.set_invoke_regularizers(true);
  SynchronizeModel(sync_args);

  std::shared_ptr<const ::artm::core::TopicModel> topic_model =
    instance_->merger()->GetLatestTopicModel(args.model_name());
  if (topic_model == nullptr) {
    LOG(ERROR) << "Unable to find " << args.model_name() << " after import";
    return;
  }

  LOG(INFO) << "Import completed, token_size = " << topic_model->token_size()
            << ", topic_size = " << topic_model->topic_size();
}

void MasterComponent::InitializeModel(const InitializeModelArgs& args) {
  instance_->merger()->InitializeModel(args);
}

void MasterComponent::Reconfigure(const MasterComponentConfig& user_config) {
  LOG(INFO) << "Merger::CreateOrReconfigureModel() with " << Helpers::Describe(user_config);
  ValidateConfig(user_config);

  MasterComponentConfig config(user_config);  // make a copy
  if (!config.has_processor_queue_max_size()) {
    // The default setting for processor queue max size is to use the number of processors.
    config.set_processor_queue_max_size(config.processors_count());
  }

  config_.set(std::make_shared<MasterComponentConfig>(config));

  if (!is_configured_) {
    // First configuration
    instance_.reset(new Instance(config));
    is_configured_ = true;
  } else {
    instance_->Reconfigure(config);
  }
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

void MasterComponent::RequestProcessBatches(const ProcessBatchesArgs& process_batches_args,
                                            ProcessBatchesResult* process_batches_result) {
  ModelName model_name = process_batches_args.pwt_source_name();
  ModelConfig model_config;
  model_config.set_name(model_name);
  model_config.set_inner_iterations_count(process_batches_args.inner_iterations_count());
  model_config.set_stream_name(process_batches_args.stream_name());
  model_config.mutable_regularizer_name()->CopyFrom(process_batches_args.regularizer_name());
  model_config.mutable_regularizer_tau()->CopyFrom(process_batches_args.regularizer_tau());
  model_config.mutable_class_id()->CopyFrom(process_batches_args.class_id());
  model_config.mutable_class_weight()->CopyFrom(process_batches_args.class_weight());

  std::shared_ptr<const TopicModel> topic_model = instance_->merger()->GetLatestTopicModel(model_name);
  std::shared_ptr<const PhiMatrix> phi_matrix = instance_->merger()->GetPhiMatrix(model_name);
  if (topic_model == nullptr && phi_matrix == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation("Model " + model_name + " does not exist"));

  const PhiMatrix& p_wt = (topic_model != nullptr) ? topic_model->GetPwt() : *phi_matrix;
  auto nwt_target(std::make_shared<DensePhiMatrix>(process_batches_args.nwt_target_name(), p_wt.topic_name()));
  nwt_target->Reshape(p_wt);
  instance_->merger()->SetPhiMatrix(process_batches_args.nwt_target_name(), nwt_target);

  model_config.set_topics_count(p_wt.topic_size());
  model_config.mutable_topic_name()->CopyFrom(p_wt.topic_name());

  BatchManager batch_manager;
  for (int batch_index = 0; batch_index < process_batches_args.batch_filename_size(); ++batch_index) {
    boost::uuids::uuid task_id = boost::uuids::random_generator()();
    batch_manager.Add(task_id, std::string(), model_name);

    auto pi = std::make_shared<ProcessorInput>();
    pi->set_notifiable(&batch_manager);
    pi->set_model_name(model_name);
    pi->set_nwt_target_name(process_batches_args.nwt_target_name());
    pi->set_batch_filename(process_batches_args.batch_filename(batch_index));
    pi->mutable_model_config()->CopyFrom(model_config);
    pi->set_task_id(task_id);
    instance_->processor_queue()->push(pi);
  }

  // ToDo - merge and extract ScoreData (require some refactoring)

  while (!batch_manager.IsEverythingProcessed()) {
    boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));
  }
}

void MasterComponent::MergeModel(const MergeModelArgs& merge_model_args) {
}

void MasterComponent::RegularizeModel(const RegularizeModelArgs& regularize_model_args) {
}

void MasterComponent::NormalizeModel(const NormalizeModelArgs& normalize_model_args) {
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

void MasterComponent::ValidateConfig(const MasterComponentConfig& config) {
  if (is_configured_) {
    std::shared_ptr<MasterComponentConfig> current_config = config_.get();
    if (current_config->disk_path() != config.disk_path()) {
      std::string message = "Changing disk_path is not supported.";
      BOOST_THROW_EXCEPTION(InvalidOperation(message));
    }
  }
}

}  // namespace core
}  // namespace artm
