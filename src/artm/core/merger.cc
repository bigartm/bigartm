// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/merger.h"

#include <algorithm>
#include <sstream>

#include "boost/lexical_cast.hpp"
#include "boost/exception/diagnostic_information.hpp"

#include "glog/logging.h"

#include "rpcz/rpc.hpp"

#include "artm/regularizer_interface.h"
#include "artm/core/call_on_destruction.h"
#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/topic_model.h"
#include "artm/core/instance_schema.h"

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

  if (ttm->topic_size() != topic_model.topics_count()) {
    std::stringstream ss;
    ss << "Unable to overwrite model '" << topic_model.name();
    ss << "' with " << topic_model.topics_count() << " topics. ";
    ss << "According to ModelConfig it must have " << ttm->topic_size() << " topics.";
    BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
  }

  bool has_classes = false;
  if (topic_model.class_id_size() != 0) {
    has_classes = true;
    if (topic_model.class_id_size() != topic_model.token_size()) {
      BOOST_THROW_EXCEPTION(InvalidOperation(
        "TopicModel.class_id_size() != TopicModel.token_size()"));
    }
  }

  bool remove_tokens = true;
  if (topic_model.token_weights_size() != 0) {
    remove_tokens = false;
    if (topic_model.token_weights_size() != topic_model.token_size()) {
      BOOST_THROW_EXCEPTION(InvalidOperation(
        "TopicModel.token_weights_size() != TopicModel.token_size()"));
    }
  }

  auto model_increment = std::make_shared<ModelIncrement>();
  model_increment->set_model_name(topic_model.name());
  model_increment->set_topics_count(topic_model.topics_count());
  for (int token_index = 0; token_index < topic_model.token_size(); ++token_index) {
    model_increment->add_token(topic_model.token(token_index));
    model_increment->add_class_id(has_classes ? topic_model.class_id(token_index) : DefaultClass);
    artm::FloatArray* token_increment = model_increment->add_token_increment();
    if (remove_tokens) {
      model_increment->add_operation_type(ModelIncrement_OperationType_DeleteToken);
    } else {
      token_increment->CopyFrom(topic_model.token_weights(token_index));
      model_increment->add_operation_type(ModelIncrement_OperationType_OverwriteValue);
    }
  }

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

void Merger::InvokePhiRegularizers(::artm::core::TopicModel* topic_model) {
  auto schema = schema_->get();
  auto& model = schema->model_config(topic_model->model_name());
  auto& reg_names = model.regularizer_name();
  auto& reg_tau = model.regularizer_tau();

  for (auto reg_name_iterator = reg_names.begin();
       reg_name_iterator != reg_names.end();
       reg_name_iterator++) {
    auto regularizer = schema->regularizer(reg_name_iterator->c_str());

    if (regularizer != nullptr) {
      auto tau_index = reg_name_iterator - reg_names.begin();
      double tau = reg_tau.Get(tau_index);

      bool retval = regularizer->RegularizePhi(topic_model, tau);
      if (!retval) {
        LOG(ERROR) << "Problems with type or number of parameters in Phi regularizer <" <<
          reg_name_iterator->c_str() <<
          ">. On this iteration this regularizer was turned off.\n";
      }
    } else {
      LOG(ERROR) << "Phi Regularizer with name <" <<
        reg_name_iterator->c_str() << "> does not exist.\n";
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

        ModelName model_name = model_increment->model_name();
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

        iter->second->ApplyDiff(*model_increment, 1.0f);
        for (int score_index = 0;
             score_index < model_increment->score_name_size();
             ++score_index) {
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

    make_rpcz_call_no_throw([&]() {
      ::artm::TopicModel reply;
      master_component_service_->RetrieveModel(request, &reply, timeout);
      std::shared_ptr< ::artm::core::TopicModel> new_global_ttm(
        new ::artm::core::TopicModel(reply));

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
    inc_ttm->second->RetrieveModelIncrement(&model_increment);
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
        *old_ttm, decay_weight, target_config == nullptr ? current_config : *target_config);
    }
    target_model_config_.set(name, nullptr);


    if (inc_ttm != topic_model_inc_.end()) {
      CuckooWatch cuckoo2("ApplyDiff, ", &cuckoo);
      new_ttm->ApplyDiff(*inc_ttm->second, apply_weight);
    }

    if (invoke_regularizers) {
      CuckooWatch cuckoo2("InvokePhiRegularizers, ", &cuckoo);
      InvokePhiRegularizers(new_ttm.get());
    }

    {
      CuckooWatch cuckoo2("CalcNormalizers, ", &cuckoo);
      new_ttm->CalcNormalizers();
    }

    {
      CuckooWatch cuckoo2("CalcPwt", &cuckoo);
      new_ttm->CalcPwt();   // calculate pwt matrix
    }

    topic_model_.set(name, new_ttm);

    topic_model_inc_.erase(name);

    // Verify if model became overregularized
    std::map<ClassId, int> degenerated_topics_count = new_ttm->FindDegeneratedTopicsCount();
    for (auto iter : degenerated_topics_count) {
      if (iter.second) {
        LOG(WARNING) << iter.second << " of " << new_ttm->topic_size()
                     << " topics have zero probability mass."
                     << " Consider reducing values of ModelConfig.regularizer_tau"
                     << " for model '" << model_name << "', class_id=" << iter.first;
      }
    }
  }
}

void Merger::InitializeModel(const InitializeModelArgs& args) {
  if (master_component_service_ != nullptr) {
    return;  // no-op in network modus operandi
  }

  auto schema = schema_->get();
  const ModelConfig& model = schema->model_config(args.model_name());
  auto new_ttm = std::make_shared< ::artm::core::TopicModel>(
      model.name(), model.topic_name());
  std::shared_ptr<DictionaryMap> dict = dictionaries_->get(args.dictionary_name());
  if (dict == nullptr) {
    std::stringstream ss;
    ss << "Dictionary " << args.dictionary_name() << " does not exist";
    BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
  }

  LOG(INFO) << "InitializeModel() with "
            << model.topics_count() << " topics and "
            << dict->size()  << " tokens";

  for (auto iter = dict->begin(); iter != dict->end(); ++iter) {
    ClassId class_id = iter->second.has_class_id() ? iter->second.class_id() : DefaultClass;
    new_ttm->AddToken(Token(class_id, iter->second.key_token()), true);
  }

  new_ttm->CalcNormalizers();
  new_ttm->CalcPwt();   // calculate pwt matrix
  topic_model_.set(args.model_name(), new_ttm);
}

}  // namespace core
}  // namespace artm
