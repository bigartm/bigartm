// Copyright 2014, Additive Regularization of Topic Models.

#include <artm/core/topic_model.h>

#include <assert.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/lexical_cast.hpp"
#include "boost/uuid/string_generator.hpp"
#include "boost/uuid/uuid_io.hpp"

#include "glog/logging.h"

#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"

namespace artm {
namespace core {

TopicModel::TopicModel(ModelName model_name, int topics_count)
    : model_name_(model_name),
      token_to_token_id_(),
      token_id_to_token_(),
      topics_count_(topics_count),
      n_wt_(),
      r_wt_(),
      n_t_(),
      n_t_default_class_(nullptr),
      batch_uuid_() {
  assert(topics_count_ > 0);
  CreateNormalizerVector(DefaultClass, topics_count_);
}

TopicModel::TopicModel(const TopicModel& rhs)
    : model_name_(rhs.model_name_),
      token_to_token_id_(rhs.token_to_token_id_),
      token_id_to_token_(rhs.token_id_to_token_),
      topics_count_(rhs.topics_count_),
      n_wt_(),  // must be deep-copied
      r_wt_(),  // must be deep-copied
      n_t_(rhs.n_t_),
      n_t_default_class_(nullptr),
      batch_uuid_(rhs.batch_uuid_) {
  for (size_t i = 0; i < rhs.n_wt_.size(); i++) {
    float* values = new float[topics_count_];
    n_wt_.push_back(values);
    memcpy(values, rhs.n_wt_[i], sizeof(float) * topics_count_);
  }

  for (size_t i = 0; i < rhs.r_wt_.size(); i++) {
    float* values = new float[topics_count_];
    r_wt_.push_back(values);
    memcpy(values, rhs.r_wt_[i], sizeof(float) * topics_count_);
  }

  n_t_default_class_ = &(n_t_.find(DefaultClass)->second);
}

TopicModel::TopicModel(const ::artm::TopicModel& external_topic_model) {
  CopyFromExternalTopicModel(external_topic_model);
}

TopicModel::TopicModel(const ::artm::core::ModelIncrement& model_increment) {
  model_name_ = model_increment.model_name();
  topics_count_ = model_increment.topics_count();

  ApplyDiff(model_increment);
}

TopicModel::~TopicModel() {
  Clear(model_name(), topic_size());
}

void TopicModel::Clear(ModelName model_name, int topics_count) {
  std::for_each(n_wt_.begin(), n_wt_.end(), [&](float* value) {
    delete [] value;
  });

  std::for_each(r_wt_.begin(), r_wt_.end(), [&](float* value) {
    delete [] value;
  });

  model_name_ = model_name;
  topics_count_ = topics_count;

  token_to_token_id_.clear();
  token_id_to_token_.clear();
  n_wt_.clear();
  r_wt_.clear();
  n_t_.clear();
  CreateNormalizerVector(DefaultClass, topics_count_);

  batch_uuid_.clear();
}

void TopicModel::RetrieveModelIncrement(::artm::core::ModelIncrement* diff) const {
  diff->set_model_name(model_name_);
  diff->set_topics_count(topic_size());

  for (int token_index = 0; token_index < token_size(); ++token_index) {
    auto current_token = token(token_index);
    diff->add_token(current_token.keyword);
    diff->add_class_id(current_token.class_id);

    ::artm::FloatArray* token_increment = diff->add_token_increment();
    for (int topic_index = 0; topic_index < topic_size(); ++topic_index) {
      token_increment->add_value(n_wt_[token_index][topic_index]);
    }
  }

  for (auto &batch : batch_uuid_) {
    diff->add_batch_uuid(boost::lexical_cast<std::string>(batch));
  }
}

void TopicModel::ApplyDiff(const ::artm::core::ModelIncrement& diff) {
  // Add new tokens discovered by processor
  for (int token_index = 0;
       token_index < diff.discovered_token_size();
       ++token_index) {
    std::string new_token = diff.discovered_token(token_index);
    ClassId new_class_id = diff.discovered_token_class_id(token_index);
    auto token = Token(new_class_id, new_token);
    if (!this->has_token(token)) {
      this->AddToken(token);
    }
  }

  int topics_count = this->topic_size();

  for (int token_index = 0;
       token_index < diff.token_increment_size();
       ++token_index) {
    const FloatArray& counters = diff.token_increment(token_index);

    auto token = Token(diff.class_id(token_index), diff.token(token_index));
    int current_token_id = token_id(token);
    if (current_token_id == -1) {
      current_token_id = this->AddToken(token, false);
    }

    for (int topic_index = 0; topic_index < topics_count; ++topic_index) {
      this->IncreaseTokenWeight(current_token_id, topic_index, counters.value(topic_index));
    }
  }

  for (int batch_index = 0;
       batch_index < diff.batch_uuid_size();
       batch_index++) {
    batch_uuid_.push_back(boost::uuids::string_generator()(diff.batch_uuid(batch_index)));
  }
}

void TopicModel::ApplyDiff(const ::artm::core::TopicModel& diff) {
  int topics_count = this->topic_size();

  for (int token_index = 0;
       token_index < diff.token_size();
       ++token_index) {
    const float* counters = diff.n_wt_[token_index];
    auto current_token = diff.token(token_index);
    if (!has_token(current_token)) {
      this->AddToken(current_token, false);
    }

    for (int topic_index = 0; topic_index < topics_count; ++topic_index) {
      this->IncreaseTokenWeight(current_token, topic_index, counters[topic_index]);
    }
  }

  for (auto &batch : diff.batch_uuid_) {
    batch_uuid_.push_back(batch);
  }
}

void TopicModel::RetrieveExternalTopicModel(::artm::TopicModel* topic_model) const {
  // 1. Fill in non-internal part of ::artm::TopicModel
  topic_model->set_name(model_name_);
  topic_model->set_topics_count(topic_size());

  for (int token_index = 0; token_index < token_size(); ++token_index) {
    auto current_token = token_id_to_token_[token_index];
    topic_model->add_token(current_token.keyword);
    topic_model->add_class_id(current_token.class_id);

    ::artm::FloatArray* weights = topic_model->add_token_weights();
    TopicWeightIterator iter = GetTopicWeightIterator(token_index);
    while (iter.NextTopic() < topic_size()) {
      weights->add_value(iter.Weight());
    }
  }

  // 2. Fill in internal part of ::artm::TopicModel
  ::artm::TopicModel_TopicModelInternals topic_model_internals;
  for (int token_index = 0; token_index < token_size(); ++token_index) {
    ::artm::FloatArray* n_wt = topic_model_internals.add_n_wt();
    ::artm::FloatArray* r_wt = topic_model_internals.add_r_wt();
    for (int topic_index = 0; topic_index < topic_size(); ++topic_index) {
      n_wt->add_value(n_wt_[token_index][topic_index]);
      r_wt->add_value(r_wt_[token_index][topic_index]);
    }
  }

  topic_model->set_internals(topic_model_internals.SerializeAsString());
}

void TopicModel::CopyFromExternalTopicModel(const ::artm::TopicModel& external_topic_model) {
  Clear(external_topic_model.name(), external_topic_model.topics_count());

  if (!external_topic_model.has_internals()) {
    // Creating a model based on weights
    for (int token_index = 0; token_index < external_topic_model.token_size(); ++token_index) {
      const std::string& token = external_topic_model.token(token_index);

      auto class_size = external_topic_model.class_id().size();
      ClassId class_id = DefaultClass;
      if (class_size == external_topic_model.token().size()) {
       class_id = external_topic_model.class_id(token_index);
      }
      int token_id = AddToken(Token(class_id, token), false);
      const ::artm::FloatArray& weights = external_topic_model.token_weights(token_index);
      for (int topic_index = 0; topic_index < topics_count_; ++topic_index) {
        SetTokenWeight(token_id, topic_index, weights.value(topic_index));
        SetRegularizerWeight(token_id, topic_index, 0);
      }
    }
  } else {
    // Creating a model based on internals
    ::artm::TopicModel_TopicModelInternals topic_model_internals;
    if (!topic_model_internals.ParseFromString(external_topic_model.internals())) {
      std::stringstream error_message;
      error_message << "Unable to deserialize internals of topic model, model_name="
                    << external_topic_model.name();
      BOOST_THROW_EXCEPTION(CorruptedMessageException(error_message.str()));
    }

    for (int token_index = 0; token_index < external_topic_model.token_size(); ++token_index) {
      const std::string& token = external_topic_model.token(token_index);
      const ClassId& class_id = external_topic_model.class_id(token_index);
      auto n_wt = topic_model_internals.n_wt(token_index);
      auto r_wt = topic_model_internals.r_wt(token_index);

      int token_id = AddToken(Token(class_id, token), false);
      for (int topic_index = 0; topic_index < topics_count_; ++topic_index) {
        SetTokenWeight(token_id, topic_index, n_wt.value(topic_index));
        SetRegularizerWeight(token_id, topic_index, r_wt.value(topic_index));
      }
    }
  }
}

int TopicModel::AddToken(ClassId class_id, std::string keyword, bool random_init) {
  return TopicModel::AddToken(Token(keyword, class_id), random_init);
}

int TopicModel::AddToken(const Token& token, bool random_init) {
  auto iter = token_to_token_id_.find(token);
  if (iter != token_to_token_id_.end()) {
    return iter->second;
  }

  int token_id = token_size();
  token_to_token_id_.insert(
      std::make_pair(token, token_id));
  token_id_to_token_.push_back(token);
  float* values = new float[topic_size()];
  n_wt_.push_back(values);

  std::vector<float>* this_class_n_t = GetNormalizerVector(token.class_id);
  if (this_class_n_t == nullptr) {
    CreateNormalizerVector(token.class_id, topic_size());
    this_class_n_t = GetNormalizerVector(token.class_id);
  }

  if (random_init) {
    float sum = 0.0f;

    for (int i = 0; i < topic_size(); ++i) {
      float val = ThreadSafeRandom::singleton().GenerateFloat();
      values[i] = val;
      sum += val;
    }

    for (int i = 0; i < topic_size(); ++i) {
      values[i] /= sum;
      (*this_class_n_t)[i] += values[i];
    }
  } else {
    memset(values, 0, sizeof(float) * topic_size());
  }

  float* regularizer_values = new float[topic_size()];
  for (int i = 0; i < topic_size(); ++i) {
    regularizer_values[i] = 0.0f;
  }

  r_wt_.push_back(regularizer_values);

  return token_id;
}

void TopicModel::IncreaseTokenWeight(const Token& token, int topic_id, float value) {
  if (!has_token(token)) {
    if (value != 0.0f) {
      LOG(ERROR) << "Token (" << token.class_id << ", " << token.keyword <<
        ") not found in the model";
    }

    return;
  }

  IncreaseTokenWeight(token_id(token), topic_id, value);
}

void TopicModel::IncreaseTokenWeight(int token_id, int topic_id, float value) {
  std::vector<float>* this_class_n_t = GetNormalizerVector(token(token_id).class_id);
  if (this_class_n_t == nullptr) {
    LOG(WARNING) << "Unknown class of token (" << token(token_id).class_id <<
      ") was found in IncreaseTokenWeight() call.";
    return;
  }

  float old_data_value = n_wt_[token_id][topic_id];
  n_wt_[token_id][topic_id] += value;

  if (old_data_value + r_wt_[token_id][topic_id] < 0) {
    if (n_wt_[token_id][topic_id] + r_wt_[token_id][topic_id] > 0) {
      (*this_class_n_t)[topic_id] += n_wt_[token_id][topic_id] + r_wt_[token_id][topic_id];
    }
  } else {
    if (n_wt_[token_id][topic_id] + r_wt_[token_id][topic_id] > 0) {
      (*this_class_n_t)[topic_id] += value;
    } else {
      (*this_class_n_t)[topic_id] -= (old_data_value + r_wt_[token_id][topic_id]);
    }
  }
}

void TopicModel::SetTokenWeight(const Token& token, int topic_id, float value) {
  if (!has_token(token)) {
    LOG(ERROR) << "Token '" << token.keyword << "' not found in the model";
    return;
  }

  SetTokenWeight(token_id(token), topic_id, value);
}

void TopicModel::SetTokenWeight(int token_id, int topic_id, float value) {
  std::vector<float>* this_class_n_t = GetNormalizerVector(token(token_id).class_id);
  if (this_class_n_t == nullptr) {
    LOG(WARNING) << "Unknown class of token (" << token(token_id).class_id <<
      ") was found in SetTokenWeight() call.";
    return;
  }

  float old_data_value = n_wt_[token_id][topic_id];
  n_wt_[token_id][topic_id] = value;

  if (old_data_value + r_wt_[token_id][topic_id] < 0) {
    if (n_wt_[token_id][topic_id] + r_wt_[token_id][topic_id] > 0) {
      (*this_class_n_t)[topic_id] += n_wt_[token_id][topic_id] + r_wt_[token_id][topic_id];
    }
  } else {
    if (n_wt_[token_id][topic_id] + r_wt_[token_id][topic_id] > 0) {
      (*this_class_n_t)[topic_id] += (n_wt_[token_id][topic_id] - old_data_value);
    } else {
      (*this_class_n_t)[topic_id] -= (old_data_value + r_wt_[token_id][topic_id]);
    }
  }
}

void TopicModel::SetRegularizerWeight(const Token& token, int topic_id, float value) {
  if (!has_token(token)) {
    LOG(ERROR) << "Token '" << token.keyword << "' not found in the model";
    return;
  }

  SetRegularizerWeight(token_id(token), topic_id, value);
}

void TopicModel::SetRegularizerWeight(int token_id, int topic_id, float value) {
  std::vector<float>* this_class_n_t = GetNormalizerVector(token(token_id).class_id);
  if (this_class_n_t == nullptr) {
    LOG(WARNING) << "Unknown class of token (" << token(token_id).class_id <<
      ") was found in SetRegularizerWeight() call.";
    return;
  }

  float old_regularizer_value = r_wt_[token_id][topic_id];
  r_wt_[token_id][topic_id] = value;

  if (n_wt_[token_id][topic_id] + old_regularizer_value < 0) {
    if (n_wt_[token_id][topic_id] + r_wt_[token_id][topic_id] > 0) {
      (*this_class_n_t)[topic_id] += n_wt_[token_id][topic_id] + r_wt_[token_id][topic_id];
    }
  } else {
    if (n_wt_[token_id][topic_id] + r_wt_[token_id][topic_id] > 0) {
      (*this_class_n_t)[topic_id] += (r_wt_[token_id][topic_id] - old_regularizer_value);
    } else {
      (*this_class_n_t)[topic_id] -= (n_wt_[token_id][topic_id] + old_regularizer_value);
    }
  }
}

void TopicModel::IncreaseRegularizerWeight(const Token& token, int topic_id, float value) {
  if (!has_token(token)) {
    if (value != 0.0f) {
      LOG(ERROR) << "Token '" << token.keyword << "' not found in the model";
    }

    return;
  }

  IncreaseRegularizerWeight(token_id(token), topic_id, value);
}

void TopicModel::IncreaseRegularizerWeight(int token_id, int topic_id, float value) {
  std::vector<float>* this_class_n_t = GetNormalizerVector(token(token_id).class_id);
  if (this_class_n_t == nullptr) {
    LOG(WARNING) << "Unknown class of token (" << token(token_id).class_id <<
      ") was found in IncreaseRegularizerWeight() call.";
    return;
  }

  float old_regularizer_value = r_wt_[token_id][topic_id];
  r_wt_[token_id][topic_id] += value;

  if (n_wt_[token_id][topic_id] + old_regularizer_value < 0) {
    if (n_wt_[token_id][topic_id] + r_wt_[token_id][topic_id] > 0) {
      (*this_class_n_t)[topic_id] += n_wt_[token_id][topic_id] + r_wt_[token_id][topic_id];
    }
  } else {
    if (n_wt_[token_id][topic_id] + r_wt_[token_id][topic_id] > 0) {
      (*this_class_n_t)[topic_id] += value;
    } else {
      (*this_class_n_t)[topic_id] -= (n_wt_[token_id][topic_id] + old_regularizer_value);
    }
  }
}

int TopicModel::token_size() const {
  return n_wt_.size();
}

int TopicModel::topic_size() const {
  return topics_count_;
}

ModelName TopicModel::model_name() const {
  return model_name_;
}

bool TopicModel::has_token(const Token& token) const {
  return token_to_token_id_.find(token) != token_to_token_id_.end();
}

int TopicModel::token_id(const Token& token) const {
  auto iter = token_to_token_id_.find(token);
  return (iter != token_to_token_id_.end()) ? iter->second : -1;
}

void TopicModel::CreateNormalizerVector(ClassId class_id, int topics_count) {
  n_t_.insert(std::pair<ClassId, std::vector<float> >(class_id,
                                                      std::vector<float>(topics_count, 0)));
  auto iter = n_t_.find(class_id);
  memset(&(iter->second[0]), 0, sizeof(float) * topics_count);
  if (class_id == DefaultClass) {
    n_t_default_class_ = &(n_t_.find(DefaultClass)->second);
  }
}

const std::vector<float>* TopicModel::GetNormalizerVector(const ClassId& class_id) const {
  return const_cast<TopicModel *>(this)->GetNormalizerVector(class_id);
}

std::vector<float>* TopicModel::GetNormalizerVector(const ClassId& class_id) {
  if (class_id == DefaultClass) {
    return n_t_default_class_;
  }

  auto iter = n_t_.find(class_id);
  if (iter == n_t_.end()) {
    return nullptr;
  }
  return &(iter->second);
}

artm::core::Token TopicModel::token(int index) const {
  assert(index >= 0);
  assert(index < token_size());
  return token_id_to_token_[index];
}

TopicWeightIterator TopicModel::GetTopicWeightIterator(
    const Token& token) const {
  auto iter = token_to_token_id_.find(token);
  assert(iter != token_to_token_id_.end());
  return std::move(TopicWeightIterator(n_wt_[iter->second], r_wt_[iter->second],
    &((*GetNormalizerVector(token.class_id))[0]), topics_count_));
}

TopicWeightIterator TopicModel::GetTopicWeightIterator(int token_id) const {
  assert(token_id >= 0);
  assert(token_id < token_size());
  return std::move(TopicWeightIterator(n_wt_[token_id], r_wt_[token_id],
    &((*GetNormalizerVector(token(token_id).class_id))[0]), topics_count_));
}

}  // namespace core
}  // namespace artm
