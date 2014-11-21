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

TopicModel::TopicModel(ModelName model_name,
    const google::protobuf::RepeatedPtrField<std::string>& topic_name)
    : model_name_(model_name),
      token_to_token_id_(),
      token_id_to_token_(),
      topic_name_(),
      n_wt_(),
      r_wt_(),
      n_t_(),
      n_t_default_class_(nullptr),
      batch_uuid_() {
  for (auto iter = topic_name.begin(); iter != topic_name.end(); ++iter) {
    topic_name_.push_back(*iter);
  }
  CreateNormalizerVector(DefaultClass, topic_size());
}

TopicModel::TopicModel(const TopicModel& rhs, float decay,
    std::shared_ptr<artm::ModelConfig> target_model_config)
    : model_name_(rhs.model_name_),
      token_to_token_id_(),
      token_id_to_token_(),
      topic_name_(rhs.topic_name_),
      n_wt_(),  // must be deep-copied
      r_wt_(),  // must be deep-copied
      n_t_(),
      n_t_default_class_(nullptr),
      batch_uuid_(rhs.batch_uuid_) {
  bool use_target_model = false;
  std::vector<bool> old_topics_mask;
  if (target_model_config != nullptr) {
    use_target_model = true;
    for (int i = 0; i < topic_size(); ++i) {
      old_topics_mask.push_back(false);
    }

    topic_name_.clear();
    for (auto& name : target_model_config->topic_name()) {
      topic_name_.push_back(name);
      for (int i = 0; i < rhs.topic_size(); ++i) {
        if (name == rhs.topic_name_[i]) {
          old_topics_mask[i] = true;
          break;
        }
      }
    }
  }
  CreateNormalizerVector(DefaultClass, topic_size());
  for (size_t token_id = 0; token_id < rhs.n_wt_.size(); token_id++) {
    AddToken(rhs.token(token_id), false);
    auto iter = rhs.GetTopicWeightIterator(token_id);
    int topic_index = 0;
    while (iter.NextTopic() < rhs.topic_size()) {
      if (use_target_model) {
        if (old_topics_mask[topic_index]) {
          SetTokenWeight(token_id, topic_index, decay * iter.NotNormalizedWeight());
          topic_index++;
        }
      } else {
        SetTokenWeight(token_id, topic_index, decay * iter.NotNormalizedWeight());
        topic_index++;
      }
    }
    if (topic_index != topic_size()) {
      // here new topics will be added into model
      float sum = 0.0f;
      std::vector<float> values;
      for (int i = topic_index; i < topic_size(); ++i) {
        float val = ThreadSafeRandom::singleton().GenerateFloat();
        values.push_back(val);
        sum += val;
      }
      for (int i = 0; i < values.size(); ++i) {
        SetTokenWeight(token_id, topic_index + i, values[i] / sum);
      }
    }
  }
}

TopicModel::TopicModel(const ::artm::TopicModel& external_topic_model) {
  CopyFromExternalTopicModel(external_topic_model);
}

TopicModel::TopicModel(const ::artm::core::ModelIncrement& model_increment) {
  model_name_ = model_increment.model_name();

  topic_name_.clear();
  auto topic_name = model_increment.topic_name();
  for (auto iter = topic_name.begin(); iter != topic_name.end(); ++iter) {
    topic_name_.push_back(*iter);
  }
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

  token_to_token_id_.clear();
  token_id_to_token_.clear();
  n_wt_.clear();
  r_wt_.clear();
  n_t_.clear();
  CreateNormalizerVector(DefaultClass, topics_count);

  batch_uuid_.clear();
}

void TopicModel::RetrieveModelIncrement(::artm::core::ModelIncrement* diff) const {
  diff->set_model_name(model_name_);
  diff->set_topics_count(topic_size());

  for (auto elem : topic_name_) {
    std::string* name = diff->add_topic_name();
    *name = elem;
  }

  for (int token_index = 0; token_index < token_size(); ++token_index) {
    auto current_token = token(token_index);
    diff->add_token(current_token.keyword);
    diff->add_class_id(current_token.class_id);
    diff->add_operation_type(ModelIncrement_OperationType_IncrementValue);

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
  int diff_token_size = diff.token_size();
  if ((diff.class_id_size() != diff_token_size) ||
      (diff.operation_type_size() != diff_token_size) ||
      (diff.token_increment_size() != diff_token_size)) {
    LOG(ERROR) << "Inconsistent fields size in ModelIncrement: "
               << diff.token_size() << " vs " << diff.class_id_size()
               << " vs " << diff.operation_type_size() << " vs " << diff.token_increment_size();
    return;
  }

  int topics_count = this->topic_size();

  for (int token_index = 0; token_index < diff_token_size; ++token_index) {
    const std::string& token_keyword = diff.token(token_index);
    const ClassId& class_id = diff.class_id(token_index);
    Token token(class_id, token_keyword);
    const FloatArray& counters = diff.token_increment(token_index);
    ModelIncrement_OperationType operation_type = diff.operation_type(token_index);
    int current_token_id = token_id(token);

    switch (operation_type) {
      case ModelIncrement_OperationType_CreateIfNotExist:
        // Add new tokens discovered by processor
        if (current_token_id == -1)
          this->AddToken(token, true);
        break;

      case ModelIncrement_OperationType_IncrementValue:
        if (current_token_id == -1)
          current_token_id = this->AddToken(token, false);
        for (int topic_index = 0; topic_index < topics_count; ++topic_index)
          this->IncreaseTokenWeight(current_token_id, topic_index, counters.value(topic_index));
        break;

      case ModelIncrement_OperationType_OverwriteValue:
        if (current_token_id == -1)
          current_token_id = this->AddToken(token, false);
        for (int topic_index = 0; topic_index < topics_count; ++topic_index)
          this->SetTokenWeight(current_token_id, topic_index, counters.value(topic_index));
        break;

      case ModelIncrement_OperationType_DeleteToken:
        this->RemoveToken(token);
        break;

      default:
        BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
          "ModelIncrement.operation_type", operation_type));
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

void TopicModel::RetrieveExternalTopicModel(
    const ::artm::GetTopicModelArgs& get_model_args,
    ::artm::TopicModel* topic_model) const {
  bool use_all_topics = false;
  bool use_all_tokens = false;
  std::vector<std::string> topics_to_use;
  std::vector<Token> tokens_to_use;
  if (get_model_args.topic_name_size() == 0) {
    use_all_topics = true;
  } else {
    for (auto name : get_model_args.topic_name()) {
      topics_to_use.push_back(name);
    }
  }
  if (get_model_args.token_size() == 0 || get_model_args.class_id_size() == 0 ||
      get_model_args.token_size() != get_model_args.class_id_size()) {
    use_all_tokens = true;
  } else {
    for (int i = 0; i < get_model_args.token_size(); ++i) {
      tokens_to_use.push_back(Token(get_model_args.class_id(i), get_model_args.token(i)));
    }
  }

  // 1. Fill in non-internal part of ::artm::TopicModel
  topic_model->set_name(model_name_);
  if (use_all_topics) {
    AddTopicsInfoInModel<std::vector<std::string> >(topic_model, topic_size(), topic_name_);
  } else {
    AddTopicsInfoInModel<google::protobuf::RepeatedPtrField<std::string> >(
        topic_model, get_model_args.topic_name_size(), get_model_args.topic_name());
  }

  for (int token_index = 0; token_index < token_size(); ++token_index) {
    auto current_token = token_id_to_token_[token_index];
    if (use_all_tokens || std::find(tokens_to_use.begin(),
                                    tokens_to_use.end(),
                                    current_token) != tokens_to_use.end()) {
      topic_model->add_token(current_token.keyword);
      topic_model->add_class_id(current_token.class_id);

      ::artm::FloatArray* weights = topic_model->add_token_weights();
      TopicWeightIterator iter = GetTopicWeightIterator(token_index);
      while (iter.NextTopic() < topic_size()) {
        if (use_all_topics || std::find(topics_to_use.begin(),
                                    topics_to_use.end(),
                                    topic_name_[iter.TopicIndex()]) != topics_to_use.end()) {
          weights->add_value(iter.Weight());
        }
      }
    }
  }

  // 2. Fill in internal part of ::artm::TopicModel
  ::artm::TopicModel_TopicModelInternals topic_model_internals;
  for (int token_index = 0; token_index < token_size(); ++token_index) {
    auto current_token = token_id_to_token_[token_index];
    if (use_all_tokens || std::find(tokens_to_use.begin(),
                                    tokens_to_use.end(),
                                    current_token) != tokens_to_use.end()) {
      ::artm::FloatArray* n_wt = topic_model_internals.add_n_wt();
      ::artm::FloatArray* r_wt = topic_model_internals.add_r_wt();
      for (int topic_index = 0; topic_index < topic_size(); ++topic_index) {
        if (use_all_topics || std::find(topics_to_use.begin(),
                                    topics_to_use.end(),
                                    topic_name_[topic_index]) != topics_to_use.end()) {
            n_wt->add_value(n_wt_[token_index][topic_index]);
            r_wt->add_value(r_wt_[token_index][topic_index]);
        }
      }
    }
  }

  topic_model->set_internals(topic_model_internals.SerializeAsString());
}

void TopicModel::CopyFromExternalTopicModel(const ::artm::TopicModel& external_topic_model) {
  Clear(external_topic_model.name(), external_topic_model.topic_name_size());

  topic_name_.clear();
  for (auto& name : external_topic_model.topic_name()) {
    topic_name_.push_back(name);
  }

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
      for (int topic_index = 0; topic_index < topic_size(); ++topic_index) {
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
      for (int topic_index = 0; topic_index < topic_size(); ++topic_index) {
        SetTokenWeight(token_id, topic_index, n_wt.value(topic_index));
        SetRegularizerWeight(token_id, topic_index, r_wt.value(topic_index));
      }
    }
  }
}

int TopicModel::AddToken(ClassId class_id, std::string keyword, bool random_init) {
  return TopicModel::AddToken(Token(class_id, keyword), random_init);
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

void TopicModel::RemoveToken(ClassId class_id, std::string keyword) {
  TopicModel::RemoveToken(Token(keyword, class_id));
}

void TopicModel::RemoveToken(const Token& token) {
  auto iter = token_to_token_id_.find(token);
  if (iter == token_to_token_id_.end())
    return;

  int token_id = iter->second;

  // Set n_wt_ and r_wt_ to zero to make sure n_t_ is still correct after token removal.
  for (int topic_id = 0; topic_id < topic_size(); ++topic_id) {
    SetTokenWeight(token_id, topic_id, 0.0f);
    SetRegularizerWeight(token_id, topic_id, 0.0f);
  }

  delete[] n_wt_[token_id];
  delete[] r_wt_[token_id];

  n_wt_.erase(n_wt_.begin() + token_id);
  r_wt_.erase(r_wt_.begin() + token_id);

  token_id_to_token_.erase(token_id_to_token_.begin() + token_id);
  token_to_token_id_.erase(iter);
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
  return topic_name_.size();
}

google::protobuf::RepeatedPtrField<std::string> TopicModel::topic_name() const {
  google::protobuf::RepeatedPtrField<std::string> topic_name;
  for (auto elem : topic_name_) {
    std::string* name = topic_name.Add();
    *name = elem;
  }
  return topic_name;
}

std::vector<ClassId> TopicModel::class_id() const {
  std::vector<ClassId> retval;
  for (auto elem : n_t_)
    retval.push_back(elem.first);
  return retval;
}

ModelName TopicModel::model_name() const {
  return model_name_;
}

bool TopicModel::has_token(const Token& token) const {
  return token_to_token_id_.count(token);
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

const artm::core::Token& TopicModel::token(int index) const {
  assert(index >= 0);
  assert(index < token_size());
  return token_id_to_token_[index];
}

int TopicModel::FindDegeneratedTopicsCount(const ClassId& class_id) const {
  const std::vector<float>* n_t = GetNormalizerVector(class_id);
  if (n_t == nullptr)
    return 0;

  int degenerated_topics_count = 0;
  for (int topic_index = 0; topic_index < n_t->size(); ++topic_index) {
    if ((*n_t)[topic_index] < 1e-20) {
      degenerated_topics_count++;
    }
  }

  return degenerated_topics_count;
}

template<typename T>
void TopicModel::AddTopicsInfoInModel(
    artm::TopicModel* topic_model, int size, const T& names) const {
  topic_model->set_topics_count(size);
  for (auto name : names) {
    std::string* blob = topic_model->add_topic_name();
    *blob = name;
  }
}

TopicWeightIterator TopicModel::GetTopicWeightIterator(
    const Token& token) const {
  auto iter = token_to_token_id_.find(token);
  assert(iter != token_to_token_id_.end());
  return std::move(TopicWeightIterator(n_wt_[iter->second], r_wt_[iter->second],
    &((*GetNormalizerVector(token.class_id))[0]), topic_size()));
}

TopicWeightIterator TopicModel::GetTopicWeightIterator(int token_id) const {
  assert(token_id >= 0);
  assert(token_id < token_size());
  return std::move(TopicWeightIterator(n_wt_[token_id], r_wt_[token_id],
    &((*GetNormalizerVector(token(token_id).class_id))[0]), topic_size()));
}

}  // namespace core
}  // namespace artm
