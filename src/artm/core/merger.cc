// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/merger.h"

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <iostream>  // NOLINT
#include <iomanip>

#include "boost/lexical_cast.hpp"
#include "boost/exception/diagnostic_information.hpp"
#include "boost/range/adaptor/map.hpp"
#include "boost/range/algorithm/copy.hpp"
#include "boost/uuid/string_generator.hpp"

#include "glog/logging.h"

#include "artm/regularizer_interface.h"
#include "artm/core/call_on_destruction.h"
#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/topic_model.h"
#include "artm/core/dense_phi_matrix.h"
#include "artm/core/phi_matrix_operations.h"
#include "artm/core/instance_schema.h"
#include "artm/core/protobuf_helpers.h"

namespace artm {
namespace core {

Merger::Merger(ThreadSafeQueue<std::shared_ptr<ModelIncrement> >* merger_queue,
               ThreadSafeHolder<InstanceSchema>* schema,
               const ::artm::core::ThreadSafeBatchCollection* batches,
               const ::artm::core::ThreadSafeDictionaryCollection* dictionaries)
    : topic_model_(),
      topic_model_inc_(),
      phi_matrix_(),
      schema_(schema),
      target_model_config_(),
      scores_merger_(),
      is_idle_(true),
      merger_queue_(merger_queue),
      batches_(batches),
      dictionaries_(dictionaries),
      is_stopping(false),
      thread_() {
  // Keep this at the last action in constructor.
  // http://stackoverflow.com/questions/15751618/initialize-boost-thread-in-object-constructor
  boost::thread t(&Merger::ThreadFunction, this);
  thread_.swap(t);
}

Merger::~Merger() {
  is_stopping = true;
  if (thread_.joinable()) {
    thread_.join();
  }
}

void Merger::DisposeModel(ModelName model_name) {
  topic_model_.erase(model_name);
  phi_matrix_.erase(model_name);
  internal_task_queue_.push(MergerTask(kDisposeModel, model_name, 0.0f, 0.0f, false, nullptr));
}

void Merger::CreateOrReconfigureModel(const ModelConfig& model) {
  if (!topic_model_.has_key(model.name())) {
    auto ttm = std::make_shared<TopicModel>(model.name(), model.topic_name());
    topic_model_.set(model.name(), ttm);
    target_model_config_.set(model.name(), nullptr);
  } else {
    target_model_config_.set(model.name(), std::make_shared<artm::ModelConfig>(model));
  }
}

void Merger::OverwriteTopicModel(const ::artm::TopicModel& topic_model) {
  auto ttm = this->GetLatestTopicModel(topic_model.name());
  if (ttm == nullptr) {
    std::stringstream ss;
    ss << "Model '" << topic_model.name();
    ss << "' can not be overwritten because it has no ModelConfig.";
    BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
  }

  auto model_increment = std::make_shared<ModelIncrement>();
  model_increment->mutable_topic_model()->CopyFrom(topic_model);
  merger_queue_->push(model_increment);
}

void Merger::ForceSynchronizeModel(const SynchronizeModelArgs& args) {
  SyncEvent sync_event;
  internal_task_queue_.push(MergerTask(kForceSynchronizeTopicModel, args.model_name(),
                            args.decay_weight(), args.apply_weight(), args.invoke_regularizers(),
                            &sync_event));
  sync_event.wait();
}

void Merger::ForceResetScores(ModelName model_name) {
  SyncEvent sync_event;
  internal_task_queue_.push(MergerTask(kForceResetScores, model_name, 0.0f, 0.0f, false, &sync_event));
  sync_event.wait();
}

std::shared_ptr<const ::artm::core::TopicModel>
Merger::GetLatestTopicModel(ModelName model_name) const {
  return topic_model_.get(model_name);
}

std::shared_ptr<const ::artm::core::PhiMatrix>
Merger::GetPhiMatrix(ModelName model_name) const {
  return phi_matrix_.get(model_name);
}

void
Merger::SetPhiMatrix(ModelName model_name, std::shared_ptr< ::artm::core::PhiMatrix> phi_matrix) {
  this->DisposeModel(model_name);
  return phi_matrix_.set(model_name, phi_matrix);
}

void Merger::ThreadFunction() {
  try {
    Helpers::SetThreadName(-1, "Merger thread");
    LOG(INFO) << "Merger thread started";
    for (;;) {
      if (is_stopping) {
        LOG(INFO) << "Merger thread stopped";
        break;
      }

      is_idle_ = true;
      boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));
      is_idle_ = false;

      for (;;) {  // MAIN FOR LOOP
        // First, handle priority tasks in the internal_task_queue.
        MergerTask merger_task;
        if (internal_task_queue_.try_pop(&merger_task)) {
          switch (merger_task.task_type) {
            case kDisposeModel:
              topic_model_inc_.erase(merger_task.model_name);
              break;
            case kForceSynchronizeTopicModel:
              SynchronizeModel(merger_task.model_name, merger_task.decay_weight,
                               merger_task.apply_weight, merger_task.invoke_regularizers);
              break;
            case kForceResetScores:
              ResetScores(merger_task.model_name);
          }

          if (merger_task.sync_event != nullptr) {
            merger_task.sync_event->signal();
          }

          continue;  // MAIN FOR LOOP
        }

        // Second, merge everything from the queue and update topic model.
        std::shared_ptr<ModelIncrement> model_increment;
        if (!merger_queue_->try_pop(&model_increment)) {
          break;  // MAIN FOR LOOP
        }

        ModelName model_name = model_increment->topic_model().name();
        auto cur_ttm = topic_model_.get(model_name);
        if (cur_ttm.get() == nullptr) {
          // model had been disposed during ongoing processing;
          continue;  // for (int model_index = 0; ...
        }

        auto iter = topic_model_inc_.find(model_name);
        if (iter == topic_model_inc_.end()) {
          topic_model_inc_.insert(std::make_pair(
            model_name, std::make_shared< ::artm::core::TopicModel>(cur_ttm->model_name(),
                                                                    cur_ttm->topic_name())));
          iter = topic_model_inc_.find(model_name);
        }

        {
          CuckooWatch cuckoo2("Merger::ApplyTopicModelOperation()");
          PhiMatrixOperations::ApplyTopicModelOperation(
            model_increment->topic_model(), 1.0f, iter->second->mutable_nwt());
        }
      }  // MAIN FOR LOOP
    }
  }
  catch(...) {
    LOG(FATAL) << boost::current_exception_diagnostic_information();
  }
}

void Merger::ResetScores(ModelName model_name) {
  LOG(INFO) << "Merger::ResetScores(" << (model_name.empty() ? "" : ("model_name=" + model_name)) << ")";
  scores_merger_.ResetScores(model_name);
}

bool Merger::RetrieveExternalTopicModel(const ::artm::GetTopicModelArgs& get_model_args,
                                        ::artm::TopicModel* topic_model) const {
  auto ttm = this->GetLatestTopicModel(get_model_args.model_name());
  if (ttm != nullptr) {
    ttm->RetrieveExternalTopicModel(get_model_args, topic_model);
    return true;
  }

  auto phi_matrix = this->GetPhiMatrix(get_model_args.model_name());
  if (phi_matrix != nullptr) {
    PhiMatrixOperations::RetrieveExternalTopicModel(*phi_matrix, get_model_args, topic_model);
    return true;
  }

  return false;
}

void Merger::RequestRegularizerState(RegularizerName regularizer_name,
                                     ::artm::RegularizerInternalState* regularizer_state) const {
  auto schema = schema_->get();
  if (schema->has_regularizer(regularizer_name)) {
    auto regularizer = schema->regularizer(regularizer_name);
    regularizer->SerializeInternalState(regularizer_state);
    regularizer_state->set_name(regularizer_name);
  } else {
    LOG(ERROR) << "Requested internal state of non-exists regularizer.";
    BOOST_THROW_EXCEPTION(InvalidOperation(
      "Attemp to request a state from non-exists regularizer"));
  }
}

bool Merger::WaitIdle(const WaitIdleArgs& args) {
  int timeout = args.timeout_milliseconds();
  auto time_start = boost::posix_time::microsec_clock::local_time();
  for (;;) {
    if (is_idle_ && merger_queue_->empty())
      break;

    boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));
    auto time_end = boost::posix_time::microsec_clock::local_time();
    if (timeout >= 0) {
      if ((time_end - time_start).total_milliseconds() >= timeout) return false;
    }
  }
  return true;
}

bool Merger::RequestScore(const GetScoreValueArgs& args,
                          ScoreData *score_data) const {
  LOG(INFO) << "Merger::RequestScore(score_name=" << args.score_name() << ")";
  std::shared_ptr<InstanceSchema> schema = schema_->get();
  if (scores_merger_.RequestScore(schema, args.model_name(), args.score_name(), score_data))
    return true;

  std::shared_ptr< ::artm::core::TopicModel> topic_model = topic_model_.get(args.model_name());
  std::shared_ptr< ::artm::core::PhiMatrix> phi_matrix = phi_matrix_.get(args.model_name());
  if (topic_model == nullptr && phi_matrix == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation("Model " + args.model_name() + " does not exist"));
  const PhiMatrix& pwt = topic_model != nullptr ? topic_model->GetPwt() : *phi_matrix;

  auto score_calculator = schema->score_calculator(args.score_name());
  if (score_calculator == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation(
      std::string("Attempt to request non-existing score: " + args.score_name())));

  if (score_calculator->is_cumulative())
    return false;

  std::shared_ptr<Score> score = score_calculator->CalculateScore(pwt);
  score_data->set_data(score->SerializeAsString());
  score_data->set_type(score_calculator->score_type());
  score_data->set_name(args.score_name());
  return true;
}

void Merger::RequestDictionary(const DictionaryName& dictionary_name, DictionaryData* dictionary_data) const {
  if (dictionaries_->has_key(dictionary_name)) {
    dictionaries_->get(dictionary_name)->StoreIntoDictionaryData(dictionary_data);
  } else {
    LOG(ERROR) << "Requested non-exists dictionary.";
    BOOST_THROW_EXCEPTION(InvalidOperation(
      "Attemp to request non-exist dictionary"));
  }
}

std::vector<ModelName> Merger::model_name() const {
  std::vector<ModelName> new_models = topic_model_.keys();
  std::vector<ModelName> old_models = phi_matrix_.keys();

  std::vector<ModelName> retval;
  retval.insert(retval.begin(), new_models.begin(), new_models.end());
  retval.insert(retval.begin(), old_models.begin(), old_models.end());
  return retval;
}

void Merger::SynchronizeModel(const ModelName& model_name, float decay_weight,
                              float apply_weight, bool invoke_regularizers) {
  std::stringstream ss;
  ss << "Merger::SynchronizeModel (" << model_name
     << ", decay_weight=" << decay_weight
     << ", apply_weight=" << apply_weight
     << ", invoke_regularizers=" << (invoke_regularizers ? "true" : "false")
     << ")";

  CuckooWatch cuckoo(ss.str());
  auto model_names = topic_model_.keys();
  if (!model_name.empty()) {
    model_names.clear();
    model_names.push_back(model_name);
  }

  for (auto &name : model_names) {
    auto old_ttm = topic_model_.get(name);
    if (old_ttm.get() == nullptr) {
      LOG(ERROR) << "Topic model " << name << " does not exist.";
      return;
    }

    std::shared_ptr<InstanceSchema> schema = schema_->get();
    if (!schema->has_model_config(name))
      return;

    const ModelConfig& current_config = schema->model_config(name);

    auto inc_ttm = topic_model_inc_.find(name);
    if (inc_ttm == topic_model_inc_.end())
      LOG(WARNING) << "SynchronizeModel() did not found any increments to topic model " << name;

    std::shared_ptr<ModelConfig> target_config = target_model_config_.get(name);

    // Accumulate counters in topic model with decay coefficient.
    std::shared_ptr< ::artm::core::TopicModel> new_ttm;
    {
      CuckooWatch cuckoo2("copy&decay, ", &cuckoo);

      new_ttm = std::make_shared< ::artm::core::TopicModel>(
        name, target_config != nullptr ? target_config->topic_name() : current_config.topic_name());

      ::artm::TopicModel topic_model;
      GetTopicModelArgs get_topic_model_args;
      get_topic_model_args.set_request_type(GetTopicModelArgs_RequestType_Nwt);
      old_ttm->RetrieveExternalTopicModel(get_topic_model_args, &topic_model);
      PhiMatrixOperations::ApplyTopicModelOperation(topic_model, decay_weight, new_ttm->mutable_nwt());
    }
    target_model_config_.set(name, nullptr);

    if (inc_ttm != topic_model_inc_.end()) {
      CuckooWatch cuckoo2("ApplyTopicModelOperation, ", &cuckoo);
      ::artm::TopicModel topic_model;
      GetTopicModelArgs get_topic_model_args;
      get_topic_model_args.set_request_type(GetTopicModelArgs_RequestType_Nwt);
      inc_ttm->second->RetrieveExternalTopicModel(get_topic_model_args, &topic_model);
      PhiMatrixOperations::ApplyTopicModelOperation(topic_model, apply_weight, new_ttm->mutable_nwt());
    }

    if (invoke_regularizers && (current_config.regularizer_settings_size() > 0)) {
      CuckooWatch cuckoo2("InvokePhiRegularizers, ", &cuckoo);

      // call CalcPwt() to allow regularizers GetPwt() usage
      new_ttm->CalcPwt();

      ::artm::core::DensePhiMatrix global_r_wt(new_ttm->model_name(), new_ttm->topic_name());
      global_r_wt.Reshape(new_ttm->GetNwt());
      PhiMatrixOperations::InvokePhiRegularizers(schema_->get(), current_config.regularizer_settings(),
                                                 new_ttm->GetPwt(), new_ttm->GetNwt(), &global_r_wt);

      // merge final r_wt with n_wt in p_wt (n_wt is const)
      new_ttm->CalcPwt(global_r_wt);

      // Verify if model became overregularized
      std::map<ClassId, std::vector<float>> new_ttm_normalizers =
        PhiMatrixOperations::FindNormalizers(new_ttm->GetNwt(), global_r_wt);
      for (auto iter : new_ttm_normalizers) {
        int bad_topics = 0;
        for (unsigned topic_index = 0; topic_index < iter.second.size(); ++topic_index) {
          if (iter.second[topic_index] < 1e-20) {
            bad_topics++;
          }
        }

        LOG_IF(WARNING, bad_topics > 0)
          << bad_topics << " of " << new_ttm->topic_size()
          << " topics have zero probability mass."
          << " Consider reducing values of ModelConfig.regularizer_tau"
          << " (or ModelConfig.regularizer_settings.tau)"
          << " for model '" << model_name << "', class_id=" << iter.first;
      }
    } else {
      CuckooWatch cuckoo2("CalcPwt", &cuckoo);
      new_ttm->CalcPwt();   // calculate pwt matrix
    }

    topic_model_.set(name, new_ttm);
    topic_model_inc_.erase(name);
  }
}

struct TokenInfo {
 public:
  TokenInfo() : num_items(0), num_total_count(0), max_one_item_weight(0) {}
  int num_items;  // number of items containing this token
  int num_total_count;  // total number of token' occurencies in the collection
  float max_one_item_weight;  // max number of token's toccurencies in one item
};

void Merger::InitializeModel(const InitializeModelArgs& args) {
  auto schema = schema_->get();
  const ModelConfig* model_config = nullptr;
  if (schema->has_model_config(args.model_name()))
    model_config = &schema->model_config(args.model_name());

  artm::TopicModel topic_model;
  topic_model.set_seed(args.seed());
  topic_model.mutable_topic_name()->CopyFrom(
    (model_config != nullptr) ? model_config->topic_name() : args.topic_name());
  topic_model.set_topics_count(topic_model.topic_name_size());

  if (args.source_type() == InitializeModelArgs_SourceType_Dictionary) {
    std::shared_ptr<Dictionary> dict = dictionaries_->get(args.dictionary_name());
    if (dict == nullptr) {
      std::stringstream ss;
      ss << "Dictionary " << args.dictionary_name() << " does not exist";
      BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    }

    LOG(INFO) << "InitializeModel() with "
      << topic_model.topic_name_size() << " topics and "
      << dict->size() << " tokens";

    for (int index = 0; index < dict->size(); ++index) {
      ClassId class_id = dict->entry(index)->has_class_id() ? dict->entry(index)->class_id() : DefaultClass;
      topic_model.add_operation_type(TopicModel_OperationType_Initialize);
      topic_model.add_class_id(class_id);
      topic_model.add_token(dict->entry(index)->key_token());
      topic_model.add_token_weights();
    }
  } else if (args.source_type() == InitializeModelArgs_SourceType_Batches) {
    std::unordered_map<Token, TokenInfo, TokenHasher> token_freq_map;
    size_t total_items_count = 0;
    float total_token_weight = 0.0f;
    std::vector<std::string> batches;
    if (args.has_disk_path()) {
      batches = BatchHelpers::ListAllBatches(args.disk_path());
      LOG(INFO) << "Found " << batches.size() << " batches in '" << args.disk_path() << "' folder";
    } else {
      for (auto& batch : args.batch_filename())
        batches.push_back(batch);
    }

    for (const std::string& batch_file : batches) {
      std::shared_ptr<Batch> batch_ptr = batches_->get(batch_file);
      if (batch_ptr == nullptr) {
        try {
          batch_ptr = std::make_shared<Batch>();
          ::artm::core::BatchHelpers::LoadMessage(batch_file, batch_ptr.get());
        }
        catch (std::exception& ex) {
          LOG(ERROR) << ex.what() << ", the batch will be skipped.";
          continue;
        }
      }

      const Batch& batch = *batch_ptr;

      std::vector<float> token_weight_in_item(batch.token_size(), 0);
      for (int item_id = 0; item_id < batch.item_size(); ++item_id) {
        total_items_count++;

        // Find cumulative weight for each token in item
        // (assume that token might have multiple occurence in each item)
        for (const Field& field : batch.item(item_id).field()) {
          for (int token_index = 0; token_index < field.token_weight_size(); ++token_index) {
            const float token_weight = field.token_weight(token_index);
            const int token_id = field.token_id(token_index);
            token_weight_in_item[token_id] += token_weight;
            total_token_weight += token_weight;
          }
        }

        for (const Field& field : batch.item(item_id).field()) {
          for (int token_index = 0; token_index < field.token_weight_size(); ++token_index) {
            const int token_id = field.token_id(token_index);
            const float token_weight = token_weight_in_item[token_id];
            if (token_weight == 0)  //  The token already had been processed -- see line (*) below
              continue;

            Token token(batch.class_id(token_id), batch.token(token_id));
            TokenInfo& token_info = token_freq_map[token];
            token_info.num_items++;
            token_info.num_total_count += token_weight;
            if (token_info.max_one_item_weight < token_weight)
              token_info.max_one_item_weight = token_weight;

            token_weight_in_item[token_id] = 0;  // (*) Makes sure each token is processed only once per item
          }
        }
      }
    }

    LOG(INFO) << "Find "
      << token_freq_map.size() << " unique tokens in "
      << total_items_count << " items, average token frequency is "
      << std::fixed << std::setw(4) << std::setprecision(5)
      << static_cast<double>(total_token_weight) / total_items_count << ".";

    for (auto& filter : args.filter()) {
      int max_freq = INT_MAX, min_freq = -1;
      int min_total_count = -1;
      float min_one_item_weight = -1;
      if (filter.has_max_percentage()) max_freq = total_items_count * filter.max_percentage();
      if (filter.has_min_percentage()) min_freq = total_items_count * filter.min_percentage();
      if (filter.has_max_items() && (max_freq > filter.max_items())) max_freq = filter.max_items();
      if (filter.has_min_items() && (min_freq < filter.min_items())) min_freq = filter.min_items();
      if (filter.has_min_total_count()) min_total_count = filter.min_total_count();
      if (filter.has_min_one_item_count()) min_one_item_weight = filter.min_one_item_count();

      for (auto iter = token_freq_map.begin(); iter != token_freq_map.end(); ++iter) {
        if (filter.has_class_id() && iter->first.class_id != filter.class_id())
          continue;
        if (iter->second.num_items > max_freq) iter->second.num_items = -1;
        if (iter->second.num_items < min_freq) iter->second.num_items = -1;
        if (iter->second.max_one_item_weight < min_one_item_weight) iter->second.num_items = -1;
        if (iter->second.num_total_count < min_total_count) iter->second.num_items = -1;
      }
    }

    size_t unique_tokens_left = 0;
    for (auto iter = token_freq_map.begin(); iter != token_freq_map.end(); ++iter) {
      if (iter->second.num_items != -1) unique_tokens_left++;
    }
    LOG(INFO) << "All filters applied, " << unique_tokens_left << " unique tokens left.";

    for (auto iter = token_freq_map.begin(); iter != token_freq_map.end(); ++iter) {
      if (iter->second.num_items != -1) {
        topic_model.add_operation_type(TopicModel_OperationType_Initialize);
        topic_model.add_class_id(iter->first.class_id);
        topic_model.add_token(iter->first.keyword);
        topic_model.add_token_weights();
      }
    }
  } else {
    BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
      "InitializeModelArgs.source_type", args.source_type()));
  }

  if (model_config != nullptr) {
    auto new_ttm = std::make_shared< ::artm::core::TopicModel>(args.model_name(), topic_model.topic_name());
    PhiMatrixOperations::ApplyTopicModelOperation(topic_model, 1.0f, new_ttm->mutable_nwt());

    new_ttm->CalcPwt();   // calculate pwt matrix
    topic_model_.set(args.model_name(), new_ttm);
  } else {
    auto new_ttm = std::make_shared< ::artm::core::DensePhiMatrix>(args.model_name(), topic_model.topic_name());
    PhiMatrixOperations::ApplyTopicModelOperation(topic_model, 1.0f, new_ttm.get());
    PhiMatrixOperations::FindPwt(*new_ttm, new_ttm.get());
    SetPhiMatrix(args.model_name(), new_ttm);
  }
}

}  // namespace core
}  // namespace artm
