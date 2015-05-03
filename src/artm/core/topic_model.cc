// Copyright 2014, Additive Regularization of Topic Models.

#include <artm/core/topic_model.h>

#include <assert.h>
#include <math.h>

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
#include "artm/core/protobuf_helpers.h"

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

TokenCollectionWeights::TokenCollectionWeights(int token_size, int topic_size)
  : topic_size_(topic_size) {
  for (int i = 0; i < token_size; ++i) {
    float* values = new float[topic_size];
    values_.push_back(values);
    memset(values, 0, sizeof(float) * topic_size);
  }
}

void TokenCollectionWeights::Reset() {
  std::for_each(values_.begin(), values_.end(), [&](float* value) {
    for (int i = 0; i < topic_size_; ++i) value[i] = 0.0f;
  });
}

void TokenCollectionWeights::Clear() {
  std::for_each(values_.begin(), values_.end(), [&](float* value) {
    delete[] value;
  });
  values_.clear();
}

int TokenCollectionWeights::AddToken() {
  float* values = new float[topic_size_];
  values_.push_back(values);
  memset(values, 0, sizeof(float) * topic_size_);
  return values_.size() - 1;
}

int TokenCollectionWeights::AddToken(const Token& token, bool random_init) {
  float* values = new float[topic_size_];
  values_.push_back(values);

  if (random_init) {
    std::vector<float> vec = Helpers::GenerateRandomVector(topic_size_, TokenHasher()(token));
    for (int i = 0; i < topic_size_; ++i) {
      values[i] = vec[i];
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

TopicModel::TopicModel(const ModelName& model_name,
    const google::protobuf::RepeatedPtrField<std::string>& topic_name)
    : model_name_(model_name),
      token_collection_(),
      topic_name_(),
      n_wt_(topic_name.size()),
      p_wt_(topic_name.size()) {
  for (auto iter = topic_name.begin(); iter != topic_name.end(); ++iter) {
    topic_name_.push_back(*iter);
  }
}

TopicModel::~TopicModel() {
  Clear(model_name(), topic_size());
}

void TopicModel::Clear(ModelName model_name, int topics_count) {
  n_wt_.Clear();
  p_wt_.Clear();
  model_name_ = model_name;
  token_collection_.Clear();
}

void TopicModel::ApplyTopicModelOperation(const ::artm::TopicModel& topic_model, float apply_weight) {
  if (!Helpers::Validate(topic_model, /* throw_error=*/ false)) return;

  const bool use_sparse_format = (topic_model.topic_index_size() > 0);
  const int this_topic_size = this->topic_size();
  std::vector<int> target_topic_index;
  if (topic_model.topic_name_size() > 0) {
    bool ok = false;
    for (auto& topic_name : topic_model.topic_name()) {
      int index = repeated_field_index_of(this->topic_name(), topic_name);
      target_topic_index.push_back(index);
      if (index != -1) ok = true;
    }
    if (!ok) {
      LOG(ERROR) << "None of TopicModel.topic_name match topic names in target model";
      return;
    }
  } else {
    if (this->topic_size() != topic_model.topics_count())
      BOOST_THROW_EXCEPTION(InvalidOperation("Mismatch between target topics_count and TopicModel.topics_count"));
    for (int i = 0; i < topic_model.topics_count(); ++i)
      target_topic_index.push_back(i);
  }

  bool optimized_execution = false;
  if ((apply_weight == 1.0f) && (target_topic_index.size() == this_topic_size)) {
    bool ok = true;
    for (int topic_index = 0; topic_index < target_topic_index.size(); ++topic_index) {
      if (target_topic_index[topic_index] != topic_index)
        ok = false;
    }
    optimized_execution = ok;
  }

  for (int token_index = 0; token_index < topic_model.token_size(); ++token_index) {
    const std::string& token_keyword = topic_model.token(token_index);
    const ClassId& class_id = topic_model.class_id(token_index);
    Token token(class_id, token_keyword);
    const FloatArray& counters = topic_model.token_weights(token_index);
    const IntArray* sparse_topic_index = use_sparse_format ? &topic_model.topic_index(token_index) : nullptr;
    const bool use_sparse_format_local = (sparse_topic_index != nullptr) && (sparse_topic_index->value_size() > 0);

    TopicModel_OperationType operation_type = topic_model.operation_type(token_index);
    int current_token_id = token_id(token);

    float* target;
    switch (operation_type) {
      case TopicModel_OperationType_Initialize:
        // Add new tokens discovered by processor
        if (current_token_id == -1)
          this->AddToken(token, true);
        break;

      case TopicModel_OperationType_Increment:
        if (current_token_id == -1)
          current_token_id = this->AddToken(token, false);
        target = n_wt_[current_token_id];

        if (optimized_execution && !use_sparse_format_local && (counters.value_size() == this_topic_size)) {
          for (int topic_index = 0; topic_index < this_topic_size; ++topic_index)
            target[topic_index] += counters.value(topic_index);
          break;
        }

        for (int i = 0; i < counters.value_size(); ++i) {
          int topic_index = use_sparse_format_local ? sparse_topic_index->value(i) : i;
          assert(topic_index < target_topic_index.size());
          if (target_topic_index[topic_index] == -1)
            continue;
          target[target_topic_index[topic_index]] += apply_weight * counters.value(i);
        }
        break;

      case TopicModel_OperationType_Overwrite:
        if (current_token_id == -1)
          current_token_id = this->AddToken(token, false);
        target = n_wt_[current_token_id];
        for (int i = 0; i < counters.value_size(); ++i) {
          int topic_index = use_sparse_format_local ? sparse_topic_index->value(i) : i;
          assert(topic_index < target_topic_index.size());
          if (target_topic_index[topic_index] == -1)
            continue;
          target[target_topic_index[topic_index]] = counters.value(i);
        }
        break;

      case TopicModel_OperationType_Remove:
        this->RemoveToken(token);
        break;

      case TopicModel_OperationType_Ignore:
        // ignore token == do nothing
        break;

      default:
        BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
          "ModelIncrement.operation_type", operation_type));
    }
  }
}

void TopicModel::RetrieveExternalTopicModel(
    const ::artm::GetTopicModelArgs& get_model_args,
    ::artm::TopicModel* topic_model) const {
  if (n_wt_.empty() && p_wt_.empty()) {
    LOG(WARNING) << "Attempt to retrieve empty topic model";
    return;
  }

  const bool use_sparse_format = get_model_args.use_sparse_format();

  std::vector<int> tokens_to_use;
  if (get_model_args.token_size() > 0) {
    bool use_default_class = (get_model_args.class_id_size() == 0);

    if (!use_default_class && (get_model_args.token_size() != get_model_args.class_id_size()))
      BOOST_THROW_EXCEPTION(artm::core::InvalidOperation(
        "GetTopicModelArgs: token_size != class_id_size, both greater then zero"));

    for (int i = 0; i < get_model_args.token_size(); ++i) {
      Token token(use_default_class ? DefaultClass : get_model_args.class_id(i),
                  get_model_args.token(i));
      int token_id = this->token_id(token);
      if (token_id != -1) {
        assert(token_id >= 0 && token_id < this->token_size());
        tokens_to_use.push_back(token_id);
      }
    }
  } else {
    if (get_model_args.class_id_size() > 0) {
      // use all tokens from the specific classes
      for (int i = 0; i < this->token_size(); ++i) {
        if (repeated_field_contains(get_model_args.class_id(), this->token(i).class_id)) {
          tokens_to_use.push_back(i);
        }
      }
    } else {
      tokens_to_use.reserve(this->token_size());
      for (int i = 0; i < this->token_size(); ++i) {
        tokens_to_use.push_back(i);
      }
    }
  }

  std::vector<int> topics_to_use;
  if (get_model_args.topic_name_size() != 0) {
    auto this_topic_name = this->topic_name();
    for (int i = 0; i < get_model_args.topic_name_size(); ++i) {
      int topic_index = repeated_field_index_of(this_topic_name, get_model_args.topic_name(i));
      if (topic_index == -1) {
        std::stringstream ss;
        ss << "GetTopicModelArgs.topic_name[" << i << "] == " << get_model_args.topic_name(i)
           << " does not exist in ModelConfig.topic_name";
        BOOST_THROW_EXCEPTION(artm::core::InvalidOperation(ss.str()));
      }

      assert(topic_index >= 0 && topic_index < this->topic_size());
      topics_to_use.push_back(topic_index);
    }
  } else {
    for (int i = 0; i < this->topic_size(); ++i)
      topics_to_use.push_back(i);
  }

  LOG(INFO) << "RetrieveExternalTopicModel() with "
            << topics_to_use.size() << " topics, "
            << tokens_to_use.size() << " tokens";

  // Populate topics_count and topic_name fields in the resulting message
  for (int topic_index : topics_to_use)
    topic_model->add_topic_name(topic_name_[topic_index]);
  topic_model->set_topics_count(topics_to_use.size());

  // Populate all non-internal part of the resulting message
  topic_model->set_name(model_name_);

  const bool use_pwt = (get_model_args.request_type() == GetTopicModelArgs_RequestType_Pwt);
  const bool use_nwt = (get_model_args.request_type() == GetTopicModelArgs_RequestType_Nwt);

  if (use_pwt && p_wt_.empty())
    BOOST_THROW_EXCEPTION(artm::core::InvalidOperation("pwt is not available in this TopicModel"));
  if (use_nwt && n_wt_.empty())
    BOOST_THROW_EXCEPTION(artm::core::InvalidOperation("nwt is not available in this TopicModel"));

  for (int token_index : tokens_to_use) {
    const Token& current_token = token_collection_.token(token_index);
    topic_model->add_token(current_token.keyword);
    topic_model->add_class_id(current_token.class_id);
    topic_model->add_operation_type(TopicModel_OperationType_Increment);

    ::artm::FloatArray *target = topic_model->add_token_weights();
    const float *source = use_pwt ? p_wt_[token_index] :
                          use_nwt ? n_wt_[token_index] : nullptr;
    if (source == nullptr)
      BOOST_THROW_EXCEPTION(artm::core::ArgumentOutOfRangeException(
        "GetTopicModelArgs.request_type", get_model_args.request_type()));

    if (!use_sparse_format) {
      target->mutable_value()->Reserve(topics_to_use.size());
      for (int topic_index : topics_to_use)
        target->add_value(source[topic_index]);
    } else {
      ::artm::IntArray* sparse_topic_index = topic_model->add_topic_index();
      for (int topics_to_use_index = 0; topics_to_use_index < topics_to_use.size(); topics_to_use_index++) {
        int topic_index = topics_to_use[topics_to_use_index];
        if (fabs(source[topic_index]) > get_model_args.eps()) {
          sparse_topic_index->add_value(topics_to_use_index);
          target->add_value(source[topic_index]);
        }
      }
    }
  }
}

int TopicModel::AddToken(const Token& token, bool random_init) {
  int token_id = token_collection_.token_id(token);
  if (token_id != -1)
    return token_id;

  token_id = token_collection_.AddToken(token);
  int token_id2 = n_wt_.AddToken(token, random_init);
  assert(token_id2 == token_id);

  return token_id;
}

void TopicModel::RemoveToken(const Token& token) {
  int token_id = token_collection_.token_id(token);
  if (token_id == -1)
    return;

  n_wt_.RemoveToken(token_id);
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

std::map<ClassId, std::vector<float> > TopicModel::FindNormalizers(const TokenCollectionWeights& r_wt_) const {
  std::map<ClassId, std::vector<float> > retval;
  for (int token_id = 0; token_id < token_size(); ++token_id) {
    const Token& token = this->token(token_id);
    auto iter = retval.find(token.class_id);
    if (iter == retval.end()) {
      retval.insert(std::pair<ClassId, std::vector<float> >(token.class_id, std::vector<float>(topic_size(), 0)));
      iter = retval.find(token.class_id);
    }

    const float* n_wt = n_wt_[token_id];
    const float* r_wt = r_wt_.empty() ? nullptr : r_wt_[token_id];
    for (int topic_id = 0; topic_id < topic_size(); ++topic_id) {
      const float sum = n_wt[topic_id] + ((r_wt == nullptr) ? 0.0f : r_wt[topic_id]);
      if (sum > 0)
        iter->second[topic_id] += sum;
    }
  }

  return retval;
}

void TopicModel::FindPwt(const TokenCollectionWeights& r_wt, TokenCollectionWeights* p_wt) const {
  const int topic_size = this->topic_size();
  const int token_size = this->token_size();

  if (topic_size == 0 || token_size == 0) {
    LOG(WARNING) << "Attempt to calculate p_wt for empty matrix";
    return;
  }

  p_wt->Clear();
  std::map<ClassId, std::vector<float> > n_t = FindNormalizers(r_wt);
  for (int token_id = 0; token_id < token_size; ++token_id) {
    const Token& token = this->token(token_id);
    int token_id2 = p_wt->AddToken(token, false);
    assert(token_id == token_id2);

    const float* nwt = n_wt_.at(token_id);
    const float* rwt = r_wt.empty() ? nullptr : r_wt.at(token_id);
    float *pwt = p_wt->at(token_id);

    const std::vector<float>& nt = n_t[token.class_id];
    for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
      if (nt[topic_index] <= 0)
        continue;

      float rwt_value = ((rwt == nullptr) ? 0.0f : rwt[topic_index]);
      float value = std::max<float>(nwt[topic_index] + rwt_value, 0.0f) / nt[topic_index];
      if (value < 1e-16) {
        // Reset small values to 0.0 to avoid performance hit.
        // http://en.wikipedia.org/wiki/Denormal_number#Performance_issues
        // http://stackoverflow.com/questions/13964606/inconsistent-multiplication-performance-with-floats
        value = 0.0f;
      }
      pwt[topic_index] = value;
    }
  }
}

TopicWeightIterator TopicModel::GetTopicWeightIterator(const Token& token) const {
  return GetTopicWeightIterator(token_collection_.token_id(token));
}

TopicWeightIterator TopicModel::GetTopicWeightIterator(int token_id) const {
  assert(token_id >= 0);
  assert(token_id < token_size());
  assert(p_wt_.size() == token_size());
  return TopicWeightIterator(p_wt_[token_id], topic_size());
}

}  // namespace core
}  // namespace artm
