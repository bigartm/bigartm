// Copyright 2015, Additive Regularization of Topic Models.

#include "artm/core/dense_phi_matrix.h"
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

DensePhiMatrix::DensePhiMatrix(const ModelName& model_name,
                               const google::protobuf::RepeatedPtrField<std::string>& topic_name)
    : model_name_(model_name), topic_name_(), token_collection_(), values_() {
  for (auto iter = topic_name.begin(); iter != topic_name.end(); ++iter) {
    topic_name_.push_back(*iter);
  }
}


void DensePhiMatrix::Reset() {
  std::for_each(values_.begin(), values_.end(), [&](float* value) {
    for (int i = 0; i < topic_size(); ++i) value[i] = 0.0f;
  });
}

void DensePhiMatrix::Clear() {
  std::for_each(values_.begin(), values_.end(), [&](float* value) {
    delete[] value;
  });
  values_.clear();
  token_collection_.Clear();
}

int DensePhiMatrix::AddToken(const Token& token, bool random_init) {
  int token_id = token_collection_.token_id(token);
  if (token_id != -1)
    return token_id;

  token_collection_.AddToken(token);

  float* values = new float[topic_size()];
  values_.push_back(values);

  if (random_init) {
    std::vector<float> vec = Helpers::GenerateRandomVector(topic_size(), TokenHasher()(token));
    for (int i = 0; i < topic_size(); ++i) {
      values[i] = vec[i];
    }
  } else {
    memset(values, 0, sizeof(float)* topic_size());
  }

  return values_.size() - 1;
}

void DensePhiMatrix::RemoveToken(const Token& token) {
  int token_id = token_collection_.token_id(token);
  if (token_id == -1)
    return;

  token_collection_.RemoveToken(token);

  if (token_id < 0 || token_id >= values_.size())
    return;

  delete[] values_[token_id];
  values_.erase(values_.begin() + token_id);
}

const Token& DensePhiMatrix::token(int index) const {
  return token_collection_.token(index);
}

google::protobuf::RepeatedPtrField<std::string> DensePhiMatrix::topic_name() const {
  google::protobuf::RepeatedPtrField<std::string> topic_name;
  for (auto elem : topic_name_) {
    std::string* name = topic_name.Add();
    *name = elem;
  }
  return topic_name;
}

std::string DensePhiMatrix::model_name() const {
  return model_name_;
}

bool DensePhiMatrix::has_token(const Token& token) const {
  return token_collection_.has_token(token);
}

int DensePhiMatrix::token_index(const Token& token) const {
  return token_collection_.token_id(token);
}

void DensePhiMatrix::Reshape(const PhiMatrix& phi_matrix) {
  Clear();
  for (int token_id = 0; token_id < phi_matrix.token_size(); ++token_id) {
    this->AddToken(phi_matrix.token(token_id), false);
  }
}

}  // namespace core
}  // namespace artm
