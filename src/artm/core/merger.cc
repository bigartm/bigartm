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

#include "glog/logging.h"

#include "rpcz/rpc.hpp"

#include "artm/regularizer_interface.h"
#include "artm/core/call_on_destruction.h"
#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/topic_model.h"
#include "artm/core/instance_schema.h"
#include "artm/core/protobuf_helpers.h"

using ::artm::core::MasterComponentService_Stub;

namespace artm {
namespace core {

Merger::Merger(ThreadSafeQueue<std::shared_ptr<ModelIncrement> >* merger_queue,
               ThreadSafeHolder<InstanceSchema>* schema,
               artm::core::MasterComponentService_Stub* master_component_service,
               const ::artm::core::ThreadSafeDictionaryCollection* dictionaries,
               Notifiable* notifiable)
    : topic_model_(),
      topic_model_inc_(),
      schema_(schema),
      target_model_config_(),
      master_component_service_(master_component_service),
      scores_merger_(schema, &topic_model_),
      is_idle_(true),
      merger_queue_(merger_queue),
      dictionaries_(dictionaries),
      notifiable_(notifiable),
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
  auto ttm = topic_model_.get(topic_model.name());
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
  rpcz::sync_event sync_event;
  internal_task_queue_.push(MergerTask(kForceSynchronizeTopicModel, args.model_name(),
                            args.decay_weight(), args.apply_weight(), args.invoke_regularizers(),
                            &sync_event));
  sync_event.wait();
}

void Merger::ForceResetScores(ModelName model_name) {
  rpcz::sync_event sync_event;
  internal_task_queue_.push(MergerTask(kForceResetScores, model_name, 0.0f, 0.0f, false, &sync_event));
  sync_event.wait();
}

void Merger::ForcePullTopicModel() {
  rpcz::sync_event sync_event;
  internal_task_queue_.push(MergerTask(kForcePullTopicModel, ModelName(), 0.0f, 0.0f, false, &sync_event));
  sync_event.wait();
}

void Merger::ForcePushTopicModelIncrement() {
  rpcz::sync_event sync_event;
  internal_task_queue_.push(MergerTask(kForcePushTopicModelIncrement, ModelName(), 0.0f, 0.0f, false, &sync_event));
  sync_event.wait();
}

std::shared_ptr<const ::artm::core::TopicModel>
Merger::GetLatestTopicModel(ModelName model_name) const {
  return topic_model_.get(model_name);
}

void Merger::InvokePhiRegularizers(const ::artm::core::TopicModel& topic_model,
                                   ::artm::core::TokenCollectionWeights* global_r_wt) {
  auto schema = schema_->get();
  auto& model = schema->model_config(topic_model.model_name());
  auto& reg_settings = model.regularizer_settings();

  int topic_size = topic_model.topic_size();
  int token_size = topic_model.token_size();

  ::artm::core::TokenCollectionWeights local_r_wt(token_size, topic_size);

  auto n_t_all = topic_model.FindNormalizers();

  for (auto reg_iterator = reg_settings.begin();
       reg_iterator != reg_settings.end();
       reg_iterator++) {
    auto regularizer = schema->regularizer(reg_iterator->name().c_str());

    if (regularizer != nullptr) {
      double tau = reg_iterator->tau();
      bool relative_reg = reg_iterator->use_relative_regularization();

      bool retval = regularizer->RegularizePhi(topic_model, &local_r_wt);

      // count n and r_i for relative regularization, if necessary
      // prepare next structure with parameters:
      // pair of pairs, first pair --- n and n_t, second one --- r_i and r_it
      std::unordered_map<core::ClassId, std::pair<std::pair<double, std::vector<float> >,
                                                  std::pair<double, std::vector<float> > > > parameters;
      std::vector<bool> topics_to_regularize;

      if (relative_reg) {
        std::vector<core::ClassId> class_ids;
        if (regularizer->class_ids_to_regularize().size() > 0) {
          auto class_ids_to_regularize = regularizer->class_ids_to_regularize();
          for (auto class_id : class_ids_to_regularize) class_ids.push_back(class_id);
        } else {
          boost::copy(n_t_all | boost::adaptors::map_keys, std::back_inserter(class_ids));
        }

        if (regularizer->topics_to_regularize().size() > 0)
          topics_to_regularize.assign(topic_size, true);
        else
          topics_to_regularize = core::is_member(topic_model.topic_name(), regularizer->topics_to_regularize());

        for (auto class_id : class_ids) {
          auto iter = n_t_all.find(class_id);
          if (iter != n_t_all.end()) {
            double n = 0.0;
            double r_i = 0.0;
            std::vector<float> r_it;
            std::vector<float> n_t = iter->second;

            for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
              if (!topics_to_regularize[topic_id]) {
                r_it.push_back(-1.0f);
                continue;
              }
              n += n_t[topic_id];

              float r_it_current = 0.0f;
              for (int token_id = 0; token_id < token_size; ++token_id) {
                if (topic_model.token(token_id).class_id != iter->first) continue;

                r_it_current += local_r_wt[token_id][topic_id];
              }

              r_it.push_back(r_it_current);
              r_i += r_it_current;
            }

            auto pair_n = std::pair<double, std::vector<float> >(n, n_t);
            auto pair_r = std::pair<double, std::vector<float> >(r_i, r_it);
            auto pair_data = std::pair<std::pair<double, std::vector<float> >,
                                       std::pair<double, std::vector<float> > >(pair_n, pair_r);
            auto pair_last = std::pair<core::ClassId,
                                    std::pair<std::pair<double, std::vector<float> >,
                                    std::pair<double, std::vector<float> > > >(iter->first, pair_data);
            parameters.insert(pair_last);
          }
        }
      }

      for (int token_id = 0; token_id < token_size; ++token_id) {
        auto iter = parameters.find(topic_model.token(token_id).class_id);
        if (relative_reg) {
          if (iter == parameters.end()) continue;
        }
        for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
          float coefficient = 1.0f;
          if (relative_reg) {
            if (!topics_to_regularize[topic_id]) continue;

            double gamma = reg_iterator->gamma();
            float n_t = iter->second.first.second[topic_id];
            float n = iter->second.first.first;
            float r_it = iter->second.second.second[topic_id];
            float r_i = iter->second.second.first;
            coefficient = static_cast<float>(gamma) * (n_t / r_it) * static_cast<float>(1 - gamma) * (n / r_i);
          }
          // update global r_wt using coefficient and tau
          (*global_r_wt)[token_id][topic_id] += coefficient * tau * local_r_wt[token_id][topic_id];
        }
      }
      local_r_wt.Reset();

      if (!retval) {
        LOG(ERROR) << "Problems with type or number of parameters in Phi regularizer <" <<
          reg_iterator->name().c_str() <<
          ">. On this iteration this regularizer was turned off.\n";
      }
    } else {
      LOG(ERROR) << "Phi Regularizer with name <" <<
        reg_iterator->name().c_str() << "> does not exist.\n";
    }
  }
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
        CuckooWatch cuckoo("Merger::MainLoopIteration", 3);

        // First, handle priority tasks in the internal_task_queue.
        MergerTask merger_task;
        if (internal_task_queue_.try_pop(&merger_task)) {
          switch (merger_task.task_type) {
            case kDisposeModel:
              topic_model_inc_.erase(merger_task.model_name);
              break;
            case kForcePullTopicModel:
              PullTopicModel();
              break;
            case kForcePushTopicModelIncrement:
              PushTopicModelIncrement();
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

        call_on_destruction c([&]() {
          if (notifiable_ != nullptr) {
            notifiable_->Callback(model_increment.get());
          }
        });

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
          CuckooWatch cuckoo2("ApplyTopicModelOperation(" +
            ((model_increment->batch_uuid_size() == 1) ? model_increment->batch_uuid(0) : "") + "), ", &cuckoo);
          iter->second->ApplyTopicModelOperation(model_increment->topic_model(), 1.0f);
        }

        for (int score_index = 0;
             score_index < model_increment->score_name_size();
             ++score_index) {
          CuckooWatch cuckoo2(std::string("AppendScore") + model_increment->score_name(score_index) + "), ", &cuckoo);
          scores_merger_.Append(model_name, model_increment->score_name(score_index),
                                model_increment->score(score_index));
        }
      }  // MAIN FOR LOOP
    }
  }
  catch(...) {
    LOG(FATAL) << boost::current_exception_diagnostic_information();
  }
}

void Merger::PullTopicModel() {
  if (master_component_service_ == nullptr) {
    return;  // no-op in local modus operandi
  }

  int timeout = schema_->get()->config().communication_timeout();

  auto model_names = topic_model_.keys();
  for (auto &model_name : model_names) {
    auto old_ttm = topic_model_.get(model_name);
    if (old_ttm.get() == nullptr)
      return;  // model had been disposed during ongoing processing;

    ::artm::GetTopicModelArgs request;
    request.set_model_name(model_name);
    request.set_request_type(GetTopicModelArgs_RequestType_Pwt);
    make_rpcz_call_no_throw([&]() {
      ::artm::TopicModel reply;
      master_component_service_->RetrieveModel(request, &reply, timeout);
      std::shared_ptr< ::artm::core::TopicModel> new_global_ttm(
        new ::artm::core::TopicModel(reply.name(), reply.topic_name()));
      new_global_ttm->ApplyTopicModelOperation(reply, 1.0f);
      new_global_ttm->CalcPwt();

      topic_model_.set(model_name, new_global_ttm);
    }, "Merger::PullTopicModel");
  }
}

void Merger::PushTopicModelIncrement() {
  if (master_component_service_ == nullptr) {
    return;  // no-op in local modus operandi
  }

  std::vector<ModelName> model_names;
  for (auto iter = topic_model_inc_.begin(); iter != topic_model_inc_.end(); ++iter) {
    model_names.push_back(iter->first);
  }

  for (auto &model_name : model_names) {
    auto inc_ttm = topic_model_inc_.find(model_name);
    if (inc_ttm == topic_model_inc_.end())
      return;  // model had been disposed during ongoing processing;

    ModelIncrement model_increment;

    ::artm::GetTopicModelArgs get_topic_model_args;
    get_topic_model_args.set_request_type(GetTopicModelArgs_RequestType_Nwt);
    inc_ttm->second->RetrieveExternalTopicModel(get_topic_model_args, model_increment.mutable_topic_model());
    scores_merger_.RetrieveModelIncrement(model_name, &model_increment);

    make_rpcz_call_no_throw([&]() {
      ::artm::core::Void reply;
      int timeout = schema_->get()->config().communication_timeout();
      master_component_service_->UpdateModel(model_increment, &reply, timeout);
      topic_model_inc_.erase(model_name);
      scores_merger_.ResetScores(model_name);
    }, "Merger::PushTopicModelIncrement");
  }
}

void Merger::ResetScores(ModelName model_name) {
  LOG(INFO) << "Merger::ResetScores(" << (model_name.empty() ? "" : ("model_name=" + model_name)) << ")";
  scores_merger_.ResetScores(model_name);
}

bool Merger::RetrieveExternalTopicModel(const ::artm::GetTopicModelArgs& get_model_args,
                                        ::artm::TopicModel* topic_model) const {
  auto ttm = this->GetLatestTopicModel(get_model_args.model_name());
  if (ttm == nullptr) return false;
  ttm->RetrieveExternalTopicModel(get_model_args, topic_model);
  return true;
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

void Merger::ScoresMerger::Append(const ModelName& model_name, const ScoreName& score_name,
                                  const std::string& score_blob) {
  auto key = std::make_pair(model_name, score_name);
  auto score_calculator = schema_->get()->score_calculator(score_name);
  if (score_calculator == nullptr) {
    LOG(ERROR) << "Unable to find score calculator: " << score_name;
    return;
  }

  auto score_inc = score_calculator->CreateScore();
  if (!score_inc->ParseFromString(score_blob)) {
    LOG(ERROR) << "Merger was unable to parse score blob. The scores might be inacurate.";
    return;
  }

  auto score = score_map_.get(key);
  if (score != nullptr) {
    score_calculator->AppendScore(*score, score_inc.get());
  }

  score_map_.set(key, score_inc);
}

void Merger::ScoresMerger::ResetScores(const ModelName& model_name) {
  if (model_name.empty()) {
    score_map_.clear();
    return;
  }

  auto keys = score_map_.keys();
  for (auto &key : keys) {
    if (key.first == model_name) {
      score_map_.erase(key);
    }
  }
}

void Merger::ScoresMerger::RetrieveModelIncrement(const ModelName& model_name,
                                                  ModelIncrement* model_increment) {
  auto keys = score_map_.keys();
  for (auto &key : keys) {
    if (key.first == model_name) {
      auto score = score_map_.get(key);
      if (score == nullptr)
        continue;

      model_increment->add_score_name(key.second);
      model_increment->add_score(score->SerializeAsString());
    }
  }
}

bool Merger::ScoresMerger::RequestScore(const GetScoreValueArgs& get_score_args,
                                        ScoreData *score_data) const {
  auto score_calculator = schema_->get()->score_calculator(get_score_args.score_name());
  if (score_calculator == nullptr) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Attempt to request non-existing score"));
  }

  if (score_calculator->is_cumulative()) {
    auto score = score_map_.get(ScoreKey(get_score_args.model_name(), get_score_args.score_name()));
    if (score == nullptr) {
      score_data->set_data(score_calculator->CreateScore()->SerializeAsString());
    } else {
      score_data->set_data(score->SerializeAsString());
    }
  } else {
    std::shared_ptr< ::artm::core::TopicModel> model = topic_model_->get(get_score_args.model_name());
    std::shared_ptr<Score> score = score_calculator->CalculateScore(*model);
    score_data->set_data(score->SerializeAsString());
  }

  score_data->set_type(score_calculator->score_type());
  score_data->set_name(get_score_args.score_name());
  return true;
}

bool Merger::RequestScore(const GetScoreValueArgs& get_score_args,
                          ScoreData *score_data) const {
  LOG(INFO) << "Merger::RequestScore(score_name=" << get_score_args.score_name() << ")";
  return scores_merger_.RequestScore(get_score_args, score_data);
}

void Merger::SynchronizeModel(const ModelName& model_name, float decay_weight,
                              float apply_weight, bool invoke_regularizers) {
  if (master_component_service_ != nullptr) {
    return;  // no-op in network modus operandi
  }

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

      if (old_ttm->token_size() > 0) {
        ::artm::TopicModel topic_model;
        GetTopicModelArgs get_topic_model_args;
        get_topic_model_args.set_request_type(GetTopicModelArgs_RequestType_Nwt);
        old_ttm->RetrieveExternalTopicModel(get_topic_model_args, &topic_model);
        new_ttm->ApplyTopicModelOperation(topic_model, decay_weight);
      }
    }
    target_model_config_.set(name, nullptr);

    if (inc_ttm != topic_model_inc_.end()) {
      CuckooWatch cuckoo2("ApplyTopicModelOperation, ", &cuckoo);
      ::artm::TopicModel topic_model;
      GetTopicModelArgs get_topic_model_args;
      get_topic_model_args.set_request_type(GetTopicModelArgs_RequestType_Nwt);
      inc_ttm->second->RetrieveExternalTopicModel(get_topic_model_args, &topic_model);
      new_ttm->ApplyTopicModelOperation(topic_model, apply_weight);
    }

    if (invoke_regularizers && (current_config.regularizer_settings_size() > 0)) {
      CuckooWatch cuckoo2("InvokePhiRegularizers, ", &cuckoo);

      // call CalcPwt() to allow regularizers GetPwt() usage
      new_ttm->CalcPwt();

      ::artm::core::TokenCollectionWeights global_r_wt(new_ttm->token_size(), new_ttm->topic_size());
      InvokePhiRegularizers(*new_ttm, &global_r_wt);

      // merge final r_wt with n_wt in p_wt (n_wt is const)
      new_ttm->CalcPwt(global_r_wt);

      // Verify if model became overregularized
      std::map<ClassId, std::vector<float>> new_ttm_normalizers = new_ttm->FindNormalizers(global_r_wt);
      for (auto iter : new_ttm_normalizers) {
        int bad_topics = 0;
        for (int topic_index = 0; topic_index < iter.second.size(); ++topic_index) {
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
  TokenInfo() : num_items(0), num_total_count(0), max_one_item_count(0) {}
  int num_items;  // number of items containing this token
  int num_total_count;  // total number of token' occurencies in the collection
  int max_one_item_count;  // max number of token's toccurencies in one item
};

void Merger::InitializeModel(const InitializeModelArgs& args) {
  if (master_component_service_ != nullptr) {
    return;  // no-op in network modus operandi
  }

  int token_duplicates = 0;

  auto schema = schema_->get();
  const ModelConfig& model = schema->model_config(args.model_name());
  auto new_ttm = std::make_shared< ::artm::core::TopicModel>(
      model.name(), model.topic_name());

  if (args.source_type() == InitializeModelArgs_SourceType_Dictionary) {
    std::shared_ptr<DictionaryMap> dict = dictionaries_->get(args.dictionary_name());
    if (dict == nullptr) {
      std::stringstream ss;
      ss << "Dictionary " << args.dictionary_name() << " does not exist";
      BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    }

    LOG(INFO) << "InitializeModel() with "
      << model.topics_count() << " topics and "
      << dict->size() << " tokens";

    for (auto iter = dict->begin(); iter != dict->end(); ++iter) {
      ClassId class_id = iter->second.has_class_id() ? iter->second.class_id() : DefaultClass;
      new_ttm->AddToken(Token(class_id, iter->second.key_token()), true);
    }
  } else if (args.source_type() == InitializeModelArgs_SourceType_Batches) {
    std::unordered_map<Token, TokenInfo, TokenHasher> token_freq_map;
    size_t total_items_count = 0, total_token_count = 0;
    std::vector<BatchManagerTask> batches = BatchHelpers::ListAllBatches(args.disk_path());
    LOG(INFO) << "Found " << batches.size() << " batches in '" << args.disk_path() << "' folder";

    for (const BatchManagerTask& batch_file : batches) {
      Batch batch;
      try {
        ::artm::core::BatchHelpers::LoadMessage(batch_file.file_path, &batch);
      }
      catch (std::exception& ex) {
        LOG(ERROR) << ex.what() << ", the batch will be skipped.";
        continue;
      }

      std::vector<char> token_mask(batch.token_size(), 0);
      for (int item_id = 0; item_id < batch.item_size(); ++item_id) {
        total_items_count++;
        total_token_count++;
        for (const Field& field : batch.item(item_id).field()) {
          for (int token_index = 0; token_index < field.token_count_size(); ++token_index) {
            const int token_count = field.token_count(token_index);
            const int token_id = field.token_id(token_index);

            total_token_count += token_count;
            if (!token_mask[token_id]) {
              token_mask[token_id] = 1;
              Token token(batch.class_id(token_id), batch.token(token_id));
              TokenInfo& token_info = token_freq_map[token];
              token_info.num_items++;
              token_info.num_total_count += token_count;
              if (token_info.max_one_item_count < token_count)
                token_info.max_one_item_count = token_count;
            } else {
              LOG_IF(WARNING, token_duplicates == 0)
                << "Token (" << batch.token(token_id) << ", " << batch.class_id(token_id)
                << ") has multiple entries in item_id=" << batch.item(item_id).id()
                << ", batch_id=" << batch.id();
              token_duplicates++;
            }
          }
        }

        for (const Field& field : batch.item(item_id).field())
          for (int token_id : field.token_id())
            token_mask[token_id] = 0;
      }
    }

    LOG(INFO) << "Find "
      << token_freq_map.size() << " unique tokens in "
      << total_items_count << " items, average token frequency is "
      << std::fixed << std::setw(4) << std::setprecision(5)
      << static_cast<double>(total_token_count) / total_items_count << ".";

    for (auto& filter : args.filter()) {
      int max_freq = INT_MAX, min_freq = -1;
      int min_total_count = -1, min_one_item_count = -1;
      if (filter.has_max_percentage()) max_freq = total_items_count * filter.max_percentage();
      if (filter.has_min_percentage()) min_freq = total_items_count * filter.min_percentage();
      if (filter.has_max_items() && (max_freq > filter.max_items())) max_freq = filter.max_items();
      if (filter.has_min_items() && (min_freq < filter.min_items())) min_freq = filter.min_items();
      if (filter.has_min_total_count()) min_total_count = filter.min_total_count();
      if (filter.has_min_one_item_count()) min_one_item_count = filter.min_one_item_count();

      for (auto iter = token_freq_map.begin(); iter != token_freq_map.end(); ++iter) {
        if (filter.has_class_id() && iter->first.class_id != filter.class_id())
          continue;
        if (iter->second.num_items > max_freq) iter->second.num_items = -1;
        if (iter->second.num_items < min_freq) iter->second.num_items = -1;
        if (iter->second.max_one_item_count < min_one_item_count) iter->second.num_items = -1;
        if (iter->second.num_total_count < min_total_count) iter->second.num_items = -1;
      }
    }

    size_t unique_tokens_left = 0;
    for (auto iter = token_freq_map.begin(); iter != token_freq_map.end(); ++iter) {
      if (iter->second.num_items != -1) unique_tokens_left++;
    }
    LOG(INFO) << "All filters applied, " << unique_tokens_left << " unique tokens left.";

    for (auto iter = token_freq_map.begin(); iter != token_freq_map.end(); ++iter)
      if (iter->second.num_items != -1)
        new_ttm->AddToken(iter->first, true);
  } else {
    BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
      "InitializeModelArgs.source_type", args.source_type()));
  }

  new_ttm->CalcPwt();   // calculate pwt matrix
  topic_model_.set(args.model_name(), new_ttm);
}

}  // namespace core
}  // namespace artm
