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
#include "artm/core/dense_phi_matrix.h"
#include "artm/core/phi_matrix_operations.h"
#include "artm/core/instance_schema.h"
#include "artm/core/protobuf_helpers.h"

namespace artm {
namespace core {

Merger::Merger(ThreadSafeHolder<InstanceSchema>* schema,
               const ::artm::core::ThreadSafeBatchCollection* batches,
               const ::artm::core::ThreadSafeDictionaryCollection* dictionaries)
    : phi_matrix_(),
      schema_(schema),
      scores_merger_(),
      batches_(batches),
      dictionaries_(dictionaries) {
}

Merger::~Merger() {
}

void Merger::DisposeModel(ModelName model_name) {
  phi_matrix_.erase(model_name);
}

void Merger::OverwriteTopicModel(const ::artm::TopicModel& topic_model) {
  auto target = std::make_shared<DensePhiMatrix>(topic_model.name(), topic_model.topic_name());
  PhiMatrixOperations::ApplyTopicModelOperation(topic_model, 1.0f, target.get());
  SetPhiMatrix(topic_model.name(), target);
}

std::shared_ptr<const ::artm::core::PhiMatrix>
Merger::GetPhiMatrix(ModelName model_name) const {
  return phi_matrix_.get(model_name);
}

std::shared_ptr<const ::artm::core::PhiMatrix>
Merger::GetPhiMatrixSafe(ModelName model_name) const {
  std::shared_ptr<const PhiMatrix> retval = phi_matrix_.get(model_name);
  if (retval == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation("Model " + model_name + " does not exist"));
  return retval;
}

void Merger::SetPhiMatrix(ModelName model_name, std::shared_ptr< ::artm::core::PhiMatrix> phi_matrix) {
  this->DisposeModel(model_name);
  return phi_matrix_.set(model_name, phi_matrix);
}

void Merger::ResetScores(ModelName model_name) {
  LOG(INFO) << "Merger::ResetScores(" << (model_name.empty() ? "" : ("model_name=" + model_name)) << ")";
  scores_merger_.ResetScores(model_name);
}

void Merger::RetrieveExternalTopicModel(const ::artm::GetTopicModelArgs& get_model_args,
                                        ::artm::TopicModel* topic_model) const {
  auto phi_matrix = this->GetPhiMatrixSafe(get_model_args.model_name());
  PhiMatrixOperations::RetrieveExternalTopicModel(*phi_matrix, get_model_args, topic_model);
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

  auto score_calculator = schema->score_calculator(args.score_name());
  if (score_calculator == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation(
      std::string("Attempt to request non-existing score: " + args.score_name())));

  if (score_calculator->is_cumulative())
    BOOST_THROW_EXCEPTION(InvalidOperation(
      "Score " + args.score_name() + " is cumulative and has not been calculated for  " + args.model_name()));

  std::shared_ptr<const ::artm::core::PhiMatrix> phi_matrix = GetPhiMatrixSafe(args.model_name());
  std::shared_ptr<Score> score = score_calculator->CalculateScore(*phi_matrix);
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
  return phi_matrix_.keys();
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

  artm::TopicModel topic_model;
  topic_model.set_seed(args.seed());
  topic_model.mutable_topic_name()->CopyFrom(args.topic_name());
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

  auto new_ttm = std::make_shared< ::artm::core::DensePhiMatrix>(args.model_name(), topic_model.topic_name());
  PhiMatrixOperations::ApplyTopicModelOperation(topic_model, 1.0f, new_ttm.get());
  PhiMatrixOperations::FindPwt(*new_ttm, new_ttm.get());
  SetPhiMatrix(args.model_name(), new_ttm);
}

}  // namespace core
}  // namespace artm
