// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_TOPIC_MODEL_H_
#define SRC_ARTM_CORE_TOPIC_MODEL_H_

#include <assert.h>

#include <algorithm>
#include <map>
#include <unordered_map>
#include <vector>
#include <set>
#include <string>

#include "boost/utility.hpp"
#include "boost/uuid/uuid.hpp"

#include "artm/messages.pb.h"

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"
#include "artm/core/phi_matrix.h"
#include "artm/utility/blas.h"

namespace artm {
namespace core {

class TopicModel;

class TokenCollection {
 public:
  void Clear();
  void RemoveToken(const Token& token);
  int  AddToken(const Token& token);

  int token_size() const;
  bool has_token(const Token& token) const;
  int token_id(const Token& token) const;
  const Token& token(int index) const;

 private:
  std::unordered_map<Token, int, TokenHasher> token_to_token_id_;
  std::vector<Token> token_id_to_token_;
};

class TokenCollectionWeights : boost::noncopyable, public PhiMatrix {
 public:
  explicit TokenCollectionWeights(int topic_size, const ::artm::core::TopicModel& parent)
      : topic_size_(topic_size), parent_(parent) {}

  TokenCollectionWeights(int token_size, int topic_size, const ::artm::core::TopicModel& parent);
  virtual ~TokenCollectionWeights() { Clear(); }

  virtual float get(int token_id, int topic_id) const { return values_[token_id][topic_id]; }
  virtual void set(int token_id, int topic_id, float value) { values_[token_id][topic_id] = value; }

  const float* operator[](int token_id) const { return values_[token_id]; }
  float* operator[](int token_id) { return values_[token_id]; }

  const float* at(int token_id) const { return values_[token_id]; }
  float* at(int token_id) { return values_[token_id]; }

  size_t size() const { return values_.size(); }
  bool empty() const { return values_.empty(); }

  virtual int topic_size() const { return topic_size_; }
  virtual int token_size() const { return values_.size(); }
  virtual const Token& token(int index) const;
  virtual bool has_token(const Token& token) const;
  virtual int token_index(const Token& token) const;
  virtual google::protobuf::RepeatedPtrField<std::string> topic_name() const;
  virtual ModelName model_name() const;

  void Reset();
  void Clear();
  int AddToken();
  int AddToken(const Token& token, bool random_init);
  void RemoveToken(int token_id);
  void Reshape(const PhiMatrix& phi_matrix);

 private:
  int topic_size_;
  std::vector<float*> values_;
  const ::artm::core::TopicModel& parent_;
};

// A class representing a topic model.
// - ::artm::core::TopicModel is an internal representation, used in Processor and Merger.
//   It supports efficient lookup of words in the matrix.
// - ::artm::TopicModel is an external representation, implemented as protobuf message.
//   It is used to transfer model back to the user.
class TopicModel {
 public:
  explicit TopicModel(const ModelName& model_name,
                      const google::protobuf::RepeatedPtrField<std::string>& topic_name);

  void Clear(ModelName model_name, int topics_count);
  virtual ~TopicModel();

  void RetrieveExternalTopicModel(
    const ::artm::GetTopicModelArgs& get_model_args,
    ::artm::TopicModel* topic_model) const;

  void ApplyTopicModelOperation(const ::artm::TopicModel& topic_model, float apply_weight);

  void RemoveToken(const Token& token);
  int  AddToken(const Token& token, bool random_init = true);

  void CalcPwt();
  void CalcPwt(const TokenCollectionWeights& r_wt);
  const TokenCollectionWeights& GetPwt() const { return p_wt_; }

  ModelName model_name() const;

  int token_size() const { return token_collection_.token_size(); }
  int topic_size() const;
  google::protobuf::RepeatedPtrField<std::string> topic_name() const;

  bool has_token(const Token& token) const { return token_collection_.has_token(token); }
  int token_id(const Token& token) const { return token_collection_.token_id(token); }
  const Token& token(int index) const { return token_collection_.token(index); }

  const TokenCollectionWeights& Nwt () const { return n_wt_; }

 private:
  ModelName model_name_;

  TokenCollection token_collection_;
  std::vector<std::string> topic_name_;

  TokenCollectionWeights n_wt_;  // vector of length tokens_count
  TokenCollectionWeights p_wt_;  // normalized matrix
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_TOPIC_MODEL_H_
