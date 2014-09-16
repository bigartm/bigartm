// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/merger.h"

#include <algorithm>

#include "boost/lexical_cast.hpp"

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

Merger::Merger(ThreadSafeQueue<std::shared_ptr<const ModelIncrement> >* merger_queue,
               ThreadSafeHolder<InstanceSchema>* schema,
               artm::core::MasterComponentService_Stub* master_component_service,
               Notifiable* notifiable)
    : topic_model_(),
      topic_model_inc_(),
      schema_(schema),
      master_component_service_(master_component_service),
      scores_merger_(schema, &topic_model_),
      is_idle_(true),
      merger_queue_(merger_queue),
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
  internal_task_queue_.push(MergerTask(kDisposeModel, model_name));
}

void Merger::CreateOrReconfigureModel(const ModelConfig& model) {
  if (!topic_model_.has_key(model.name())) {
    // Handle more type of reconfigs - for example, changing the number of topics;
    auto ttm = std::make_shared<TopicModel>(model.name(), model.topics_count());
    topic_model_.set(model.name(), ttm);
  }

  auto ttm = topic_model_.get(model.name());
  if (ttm->topic_size() != model.topics_count()) {
    std::string message("Unable to change the number of topics in topic model");
    BOOST_THROW_EXCEPTION(InvalidOperation(message));
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

  std::shared_ptr<::artm::core::TopicModel> new_ttm(new ::artm::core::TopicModel(topic_model));
  topic_model_.set(topic_model.name(), new_ttm);
}

void Merger::ForceResetScores(ModelName model_name) {
  rpcz::sync_event sync_event;
  internal_task_queue_.push(MergerTask(kForceResetScores, model_name, &sync_event));
  sync_event.wait();
}

void Merger::ForcePullTopicModel() {
  rpcz::sync_event sync_event;
  internal_task_queue_.push(MergerTask(kForcePullTopicModel, ModelName(), &sync_event));
  sync_event.wait();
}

void Merger::ForcePushTopicModelIncrement() {
  rpcz::sync_event sync_event;
  internal_task_queue_.push(MergerTask(kForcePushTopicModelIncrement, ModelName(), &sync_event));
  sync_event.wait();
}

std::shared_ptr<const ::artm::core::TopicModel>
Merger::GetLatestTopicModel(ModelName model_name) const {
  return topic_model_.get(model_name);
}

void Merger::InvokePhiRegularizers() {
  auto schema = schema_->get();
  std::vector<ModelName> model_names = schema->GetModelNames();

  std::for_each(model_names.begin(), model_names.end(), [&](ModelName model_name) {
    const ModelConfig& model = schema->model_config(model_name);
    auto cur_ttm = topic_model_.get(model_name);

    if (cur_ttm.get() != nullptr) {
      auto reg_names = model.regularizer_name();
      auto reg_tau = model.regularizer_tau();
      auto new_ttm = std::make_shared<::artm::core::TopicModel>(*cur_ttm);

      for (auto reg_name_iterator = reg_names.begin(); reg_name_iterator != reg_names.end();
        reg_name_iterator++) {
        auto regularizer = schema->regularizer(reg_name_iterator->c_str());

        if (regularizer != nullptr) {
          auto tau_index = reg_name_iterator - reg_names.begin();
          double tau = reg_tau.Get(tau_index);

          bool retval = regularizer->RegularizePhi(new_ttm.get(), tau);
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
      topic_model_.set(model_name, new_ttm);
    }
  });
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

      // Sleep and check for interrupt.
      // To check for interrupt without sleep,
      // use boost::this_thread::interruption_point()
      // which also throws boost::thread_interrupted
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
            case kForceResetScores:
              ResetScores(merger_task.model_name);
          }

          if (merger_task.sync_event != nullptr) {
            merger_task.sync_event->signal();
          }

          continue;  // MAIN FOR LOOP
        }

        // Second, merge everything from the queue and update topic model.
        std::shared_ptr<const ModelIncrement> model_increment;
        if (!merger_queue_->try_pop(&model_increment)) {
          break;  // MAIN FOR LOOP
        }

        call_on_destruction c([&]() {
          if (notifiable_ != nullptr) {
            notifiable_->Callback(model_increment);
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
            model_name, std::make_shared<::artm::core::TopicModel>(cur_ttm->model_name(),
                                                                   cur_ttm->topic_size())));
          iter = topic_model_inc_.find(model_name);
        }

        iter->second->ApplyDiff(*model_increment);
        for (int score_index = 0;
             score_index < model_increment->score_name_size();
             ++score_index) {
          scores_merger_.Append(model_name, model_increment->score_name(score_index),
                                model_increment->score(score_index));
        }
      }  // MAIN FOR LOOP
    }
  }
  catch(boost::thread_interrupted&) {
    LOG(WARNING) << "thread_interrupted exception in Merger::ThreadFunction() function";
    return;
  } catch(...) {
    LOG(FATAL) << "Fatal exception in Merger::ThreadFunction() function";
    throw;
  }
}

void Merger::PullTopicModel() {
  auto model_names = topic_model_.keys();
  for (auto &model_name : model_names) {
    auto old_ttm = topic_model_.get(model_name);
    if (old_ttm.get() == nullptr)
      return;  // model had been disposed during ongoing processing;

    if (master_component_service_ == nullptr) {
      auto inc_ttm = topic_model_inc_.find(model_name);
      if (inc_ttm == topic_model_inc_.end())
       return;  // model had been disposed during ongoing processing;

      // Old mode: accumulate counters in topic model forever
      // auto new_ttm = std::make_shared<::artm::core::TopicModel>(*old_ttm);

      // New mode: accumulate counters only accross one iteration, then re-calculate Phi from scratch.
      auto new_ttm = std::make_shared<::artm::core::TopicModel>(old_ttm->model_name(),
                                                                old_ttm->topic_size());
      new_ttm->ApplyDiff(*inc_ttm->second);
      topic_model_.set(model_name, new_ttm);
      topic_model_inc_.erase(model_name);
    } else {
      try {
        ::artm::core::String request;
        request.set_value(model_name);
        ::artm::TopicModel reply;
        master_component_service_->RetrieveModel(request, &reply);
        std::shared_ptr<::artm::core::TopicModel> new_global_ttm(
          new ::artm::core::TopicModel(reply));

        topic_model_.set(model_name, new_global_ttm);
        topic_model_inc_.erase(model_name);
      } catch(const rpcz::rpc_error&) {
        LOG(ERROR) << "Merger failed to pull topic model from the master component service.";
        throw;
      }
    }
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

    try {
      ::artm::core::Void reply;
      master_component_service_->UpdateModel(model_increment, &reply);
      topic_model_inc_.erase(model_name);
      scores_merger_.ResetScores(model_name);
    } catch(const rpcz::rpc_error&) {
      LOG(ERROR) << "Merger failed to send updates to master component service.";
      throw;
    }
  }
}

void Merger::ResetScores(ModelName model_name) {
  scores_merger_.ResetScores(model_name);
}

bool Merger::RetrieveExternalTopicModel(ModelName model_name,
                                        ::artm::TopicModel* topic_model) const {
  auto ttm = this->GetLatestTopicModel(model_name);
  if (ttm == nullptr) return false;
  ttm->RetrieveExternalTopicModel(topic_model);
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

bool Merger::WaitIdle(int timeout) {
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

bool Merger::ScoresMerger::RequestScore(const ModelName& model_name, const ScoreName& score_name,
                                        ScoreData *score_data) const {
  auto score_calculator = schema_->get()->score_calculator(score_name);
  if (score_calculator == nullptr) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Attempt to request non-existing score"));
  }

  if (score_calculator->is_cumulative()) {
    auto score = score_map_.get(ScoreKey(model_name, score_name));
    if (score == nullptr) {
      score_data->set_data(score_calculator->CreateScore()->SerializeAsString());
    } else {
      score_data->set_data(score->SerializeAsString());
    }
  } else {
    std::shared_ptr<::artm::core::TopicModel> model = topic_model_->get(model_name);
    std::shared_ptr<Score> score = score_calculator->CalculateScore(*model);
    score_data->set_data(score->SerializeAsString());
  }

  score_data->set_type(score_calculator->score_type());
  score_data->set_name(score_name);
  return true;
}

bool Merger::RequestScore(const ModelName& model_name, const ScoreName& score_name,
                          ScoreData *score_data) const {
  return scores_merger_.RequestScore(model_name, score_name, score_data);
}

}  // namespace core
}  // namespace artm

