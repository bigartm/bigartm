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

void TokenCollection::RemoveToken(const Token& token) {
  auto iter = token_to_token_id_.find(token);
  if (iter == token_to_token_id_.end())
    return;

  int token_id = iter->second;
  token_id_to_token_.erase(token_id_to_token_.begin() + token_id);
  token_to_token_id_.erase(iter);
}

int TokenCollection::AddToken(const Token& token) {
  int token_id = this->token_id(token);
  if (token_id != -1)
    return token_id;

  token_id = token_size();
  token_to_token_id_.insert(
    std::make_pair(token, token_id));
  token_id_to_token_.push_back(token);
  return token_id;
}

bool TokenCollection::has_token(const Token& token) const {
  return token_to_token_id_.count(token);
}

int TokenCollection::token_id(const Token& token) const {
  auto iter = token_to_token_id_.find(token);
  return (iter != token_to_token_id_.end()) ? iter->second : -1;
}

const Token& TokenCollection::token(int index) const {
  return token_id_to_token_[index];
}

void TokenCollection::Clear() {
  token_to_token_id_.clear();
  token_id_to_token_.clear();
}

int TokenCollection::token_size() const {
  return token_to_token_id_.size();
}

void TokenCollectionWeights::Clear() {
  std::for_each(values_.begin(), values_.end(), [&](float* value) {
    delete[] value;
  });
  values_.clear();
}

int TokenCollectionWeights::AddToken(bool random_init) {
  float* values = new float[topic_size_];
  values_.push_back(values);

  if (random_init) {
    for (int i = 0; i < topic_size_; ++i) {
      values[i] = ThreadSafeRandom::singleton().GenerateFloat();
    }
  } else {
    memset(values, 0, sizeof(float)* topic_size_);
  }

  return values_.size() - 1;
}

void TokenCollectionWeights::RemoveToken(int token_id) {
  if (token_id < 0 || token_id >= values_.size())
    return;

  delete[] values_[token_id];
  values_.erase(values_.begin() + token_id);
}

TopicModel::TopicModel(ModelName model_name,
    const google::protobuf::RepeatedPtrField<std::string>& topic_name)
    : model_name_(model_name),
      token_collection_(),
      topic_name_(),
      n_wt_(topic_name.size()),
      r_wt_(topic_name.size()),
      n_t_(),
      n_t_default_class_(nullptr),
      batch_uuid_() {
  for (auto iter = topic_name.begin(); iter != topic_name.end(); ++iter) {
    topic_name_.push_back(*iter);
  }
}

TopicModel::TopicModel(const TopicModel& rhs, float decay,
                       const artm::ModelConfig& target_model_config)
    : model_name_(rhs.model_name_),
      token_collection_(),
      topic_name_(rhs.topic_name_),
      n_wt_(target_model_config.topics_count()),  // must be deep-copied
      r_wt_(target_model_config.topics_count()),  // must be deep-copied
      n_t_(),
      n_t_default_class_(nullptr),
      batch_uuid_(rhs.batch_uuid_) {
  std::vector<bool> old_topics_mask;
  for (int i = 0; i < topic_size(); ++i) {
    old_topics_mask.push_back(false);
  }

  topic_name_.clear();
  for (auto& name : target_model_config.topic_name()) {
    topic_name_.push_back(name);
    for (int i = 0; i < rhs.topic_size(); ++i) {
      if (name == rhs.topic_name_[i]) {
        old_topics_mask[i] = true;
        break;
      }
    }
  }

  for (size_t token_id = 0; token_id < rhs.token_size(); token_id++) {
    AddToken(rhs.token(token_id), false);
    auto iter = rhs.GetTopicWeightIterator(token_id);
    int topic_index = 0;
    while (iter.NextTopic() < rhs.topic_size()) {
      if (old_topics_mask[topic_index]) {
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

TopicModel::TopicModel(const ::artm::TopicModel& external_topic_model)
    : n_wt_(external_topic_model.topics_count()),
      r_wt_(external_topic_model.topics_count()) {
  CopyFromExternalTopicModel(external_topic_model);
}

TopicModel::TopicModel(const ::artm::core::ModelIncrement& model_increment)
    : n_wt_(model_increment.topic_name_size()),
      r_wt_(model_increment.topic_name_size()) {
  model_name_ = model_increment.model_name();

  topic_name_.clear();
  auto topic_name = model_increment.topic_name();
  for (auto iter = topic_name.begin(); iter != topic_name.end(); ++iter) {
    topic_name_.push_back(*iter);
  }
  ApplyDiff(model_increment, 1.0f);
}

TopicModel::~TopicModel() {
  Clear(model_name(), topic_size());
}

void TopicModel::Clear(ModelName model_name, int topics_count) {
  n_wt_.Clear();
  r_wt_.Clear();
  model_name_ = model_name;
  token_collection_.Clear();
  batch_uuid_.clear();
  n_t_.clear();
  n_t_default_class_ = nullptr;
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

void TopicModel::ApplyDiff(const ::artm::core::ModelIncrement& diff, float apply_weight) {
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

    float* target;
    switch (operation_type) {
      case ModelIncrement_OperationType_CreateIfNotExist:
        // Add new tokens discovered by processor
        if (current_token_id == -1)
          this->AddToken(token, true);
        break;

      case ModelIncrement_OperationType_IncrementValue:
        if (counters.value_size() == 0)
          break;

        if (counters.value_size() != topics_count) {
          LOG(ERROR) << "ModelIncrement_OperationType_IncrementValue: counters.value_size() != topics_count";
          break;
        }

        if (current_token_id == -1)
          current_token_id = this->AddToken(token, false);
        target = n_wt_[current_token_id];
        for (int topic_index = 0; topic_index < topics_count; ++topic_index)
          target[topic_index] += apply_weight * counters.value(topic_index);
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

      case ModelIncrement_OperationType_SkipToken:
        // skip token == do nothing
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

void TopicModel::ApplyDiff(const ::artm::core::TopicModel& diff, float apply_weight) {
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
      this->IncreaseTokenWeight(current_token, topic_index, apply_weight * counters[topic_index]);
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
  std::vector<ClassId> class_ids_to_use;
  std::vector<std::string> topics_to_use;
  std::vector<Token> tokens_to_use;

  if (get_model_args.topic_name_size() == 0) {
    use_all_topics = true;
  } else {
    for (auto name : get_model_args.topic_name()) {
      topics_to_use.push_back(name);
    }
  }

  int args_class_id_size = get_model_args.class_id_size();
  int args_token_size = get_model_args.token_size();
  if (args_class_id_size == 0) {
    use_all_tokens = true;
  } else {
    if (args_token_size != 0) {
      if (args_token_size != args_class_id_size) {
        BOOST_THROW_EXCEPTION(artm::core::InvalidOperation(
            "GetTopicModelArgs: token_size != class_id_size, both greater then zero"));
      } else {
        for (int i = 0; i < args_token_size; ++i) {
          tokens_to_use.push_back(Token(get_model_args.class_id(i), get_model_args.token(i)));
        }
      }
    } else {
      for (int i = 0; i < args_class_id_size; ++i) {
        class_ids_to_use.push_back(get_model_args.class_id(i));
      }
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
    const Token& current_token = token_collection_.token(token_index);
    if (use_all_tokens ||
        std::find(tokens_to_use.begin(),
                  tokens_to_use.end(),
                  current_token) != tokens_to_use.end() ||
        std::find(class_ids_to_use.begin(),
                  class_ids_to_use.end(),
                  current_token.class_id) != class_ids_to_use.end()) {
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
    const Token& current_token = token_collection_.token(token_index);
    if (use_all_tokens ||
        std::find(tokens_to_use.begin(),
                  tokens_to_use.end(),
                  current_token) != tokens_to_use.end() ||
        std::find(class_ids_to_use.begin(),
                  class_ids_to_use.end(),
                  current_token.class_id) != class_ids_to_use.end()) {
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

  CalcNormalizers();
  CalcPwt();
}

int TopicModel::AddToken(const Token& token, bool random_init) {
  int token_id = token_collection_.token_id(token);
  if (token_id != -1)
    return token_id;

  token_id = token_collection_.AddToken(token);
  int token_id2 = n_wt_.AddToken(random_init);
  assert(token_id2 == token_id);

  int token_id3 = r_wt_.AddToken(false);
  assert(token_id3 == token_id);

  return token_id;
}

void TopicModel::RemoveToken(const Token& token) {
  int token_id = token_collection_.token_id(token);
  if (token_id == -1)
    return;

  n_wt_.RemoveToken(token_id);
  r_wt_.RemoveToken(token_id);
  token_collection_.RemoveToken(token);
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
  n_wt_[token_id][topic_id] += value;
}

void TopicModel::SetTokenWeight(const Token& token, int topic_id, float value) {
  if (!has_token(token)) {
    LOG(ERROR) << "Token '" << token.keyword << "' not found in the model";
    return;
  }

  SetTokenWeight(token_id(token), topic_id, value);
}

void TopicModel::SetTokenWeight(int token_id, int topic_id, float value) {
  n_wt_[token_id][topic_id] = value;
}

void TopicModel::SetRegularizerWeight(const Token& token, int topic_id, float value) {
  if (!has_token(token)) {
    LOG(ERROR) << "Token '" << token.keyword << "' not found in the model";
    return;
  }

  SetRegularizerWeight(token_id(token), topic_id, value);
}

void TopicModel::SetRegularizerWeight(int token_id, int topic_id, float value) {
  r_wt_[token_id][topic_id] = value;
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
  r_wt_[token_id][topic_id] += value;
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

ModelName TopicModel::model_name() const {
  return model_name_;
}

void TopicModel::CalcNormalizers() {
  n_t_.clear();
  n_t_default_class_ = nullptr;
  for (int token_id = 0; token_id < token_size(); ++token_id) {
    const Token& token = this->token(token_id);
    std::vector<float>* n_t = GetNormalizerVector(token.class_id);
    if (n_t == nullptr) {
      n_t = CreateNormalizerVector(token.class_id, topic_size());
    }
    float* n_wt = n_wt_[token_id];
    float* r_wt = r_wt_[token_id];
    for (int topic_id = 0; topic_id < topic_size(); ++topic_id) {
      float sum = n_wt[topic_id] + r_wt[topic_id];
      if (sum > 0)
        (*n_t)[topic_id] += sum;
    }
  }
}

void TopicModel::CalcPwt() {
  const int topic_size = this->topic_size();
  const int token_size = this->token_size();
  p_wt_.reset(new ::artm::utility::DenseMatrix<float>(token_size, topic_size));
  p_wt_->InitializeZeros();

  for (int token_id = 0; token_id < token_size; ++token_id) {
    const Token& token = this->token(token_id);
    auto topic_iter = this->GetTopicWeightIterator(token);
    for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
      float value = topic_iter[topic_index];
      if (value < 1e-16) {
        // Reset small values to 0.0 to avoid performance hit.
        // http://en.wikipedia.org/wiki/Denormal_number#Performance_issues
        // http://stackoverflow.com/questions/13964606/inconsistent-multiplication-performance-with-floats
        value = 0.0f;
      }
      (*p_wt_)(token_id, topic_index) = value;
    }
  }
}

std::vector<float>* TopicModel::CreateNormalizerVector(ClassId class_id, int topics_count) {
  n_t_.insert(std::pair<ClassId, std::vector<float> >(class_id,
                                                      std::vector<float>(topics_count, 0)));
  auto iter = n_t_.find(class_id);
  memset(&(iter->second[0]), 0, sizeof(float) * topics_count);
  if (class_id == DefaultClass) {
    n_t_default_class_ = &(n_t_.find(DefaultClass)->second);
  }

  return GetNormalizerVector(class_id);
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

std::map<ClassId, int> TopicModel::FindDegeneratedTopicsCount() const {
  std::map<ClassId, int> retval;

  for (int token_id = 0; token_id < token_size(); ++token_id) {
    ClassId class_id = token(token_id).class_id;
    if (retval.find(class_id) != retval.end())
      continue;

    const std::vector<float>* n_t = GetNormalizerVector(class_id);
    if (n_t == nullptr)
      continue;

    int degenerated_topics_count = 0;
    for (int topic_index = 0; topic_index < n_t->size(); ++topic_index) {
      if ((*n_t)[topic_index] < 1e-20) {
        degenerated_topics_count++;
      }
    }

    retval.insert(std::make_pair(class_id, degenerated_topics_count));
  }

  return retval;
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
  int token_id = token_collection_.token_id(token);
  assert(token_id != -1);
  return std::move(TopicWeightIterator(n_wt_[token_id], r_wt_[token_id],
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
