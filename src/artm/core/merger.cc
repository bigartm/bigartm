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

void Merger::OverwriteTopicModel(const ::artm::TopicModel& topic_model) {
  auto ttm = this->GetLatestTopicModel(topic_model.name());
  if (ttm != nullptr) {
    // Create old-style model
    auto model_increment = std::make_shared<ModelIncrement>();
    model_increment->mutable_topic_model()->CopyFrom(topic_model);
    merger_queue_->push(model_increment);
    return;
  }

  // Create new-style model
  auto target = std::make_shared<DensePhiMatrix>(topic_model.name(), topic_model.topic_name());
  PhiMatrixOperations::ApplyTopicModelOperation(topic_model, 1.0f, target.get());
  SetPhiMatrix(topic_model.name(), target);
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

void Merger::SetPhiMatrix(ModelName model_name, std::shared_ptr< ::artm::core::PhiMatrix> phi_matrix) {
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
          VLOG(0) << "MasterComponent: start merging processor increments";
          PhiMatrixOperations::ApplyTopicModelOperation(
            model_increment->topic_model(), 1.0f, iter->second->mutable_nwt());
          VLOG(0) << "MasterComponent: complete merging processor increments";
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

void Merger::RetrieveExternalTopicModel(const ::artm::GetTopicModelArgs& get_model_args,
                                        ::artm::TopicModel* topic_model) const {
  auto ttm = this->GetLatestTopicModel(get_model_args.model_name());
  auto phi_matrix = this->GetPhiMatrix(get_model_args.model_name());
  if (ttm == nullptr && phi_matrix == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation("Model " + get_model_args.model_name() + " does not exist"));
  if (ttm != nullptr) {
    ttm->RetrieveExternalTopicModel(get_model_args, topic_model);
  } else {
    PhiMatrixOperations::RetrieveExternalTopicModel(*phi_matrix, get_model_args, topic_model);
  }
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

void Merger::RequestScore(const GetScoreValueArgs& args,
                          ScoreData *score_data) const {
  LOG(INFO) << "Merger::RequestScore(score_name=" << args.score_name() << ")";
  std::shared_ptr<InstanceSchema> schema = schema_->get();
  if (scores_merger_.RequestScore(schema, args.model_name(), args.score_name(), score_data))
    return;  // success

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
    BOOST_THROW_EXCEPTION(InvalidOperation(
      "Score " + args.score_name() + " is cumulative and has not been calculated for  " + args.model_name()));

  std::shared_ptr<Score> score = score_calculator->CalculateScore(pwt);
  score_data->set_data(score->SerializeAsString());
  score_data->set_type(score_calculator->score_type());
  score_data->set_name(args.score_name());
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

  std::shared_ptr<Dictionary> dict = dictionaries_->get(args.dictionary_name());
  if (dict == nullptr) {
    std::stringstream ss;
    ss << "Dictionary '" << args.dictionary_name() << "' does not exist";
    BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
  }

  if (dict->size() == 0) {
    std::stringstream ss;
    ss << "Dictionary '" << args.dictionary_name() << "' has no entries";
    BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
  }

  LOG(INFO) << "InitializeModel() with "
    << topic_model.topic_name_size() << " topics and "
    << dict->size() << " tokens";

  for (int index = 0; index < dict->size(); ++index) {
    topic_model.add_operation_type(TopicModel_OperationType_Initialize);
    topic_model.add_class_id(dict->entry(index)->token().class_id);
    topic_model.add_token(dict->entry(index)->token().keyword);
    topic_model.add_token_weights();
  }

  if (model_config != nullptr) {
    auto new_ttm = std::make_shared< ::artm::core::TopicModel>(args.model_name(), topic_model.topic_name());
    PhiMatrixOperations::ApplyTopicModelOperation(topic_model, 1.0f, new_ttm->mutable_nwt());
    PhiMatrixOperations::FindPwt(new_ttm->GetNwt(), new_ttm->mutable_nwt());
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
