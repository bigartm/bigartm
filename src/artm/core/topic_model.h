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
#include "artm/core/regularizable.h"
#include "artm/utility/blas.h"

namespace artm {
namespace core {

class TopicModel;

// A class representing an iterator over one row from Phi matrix
// (a vector of topic weights for a particular token of the topic model).
// Remark: currently this method iterates over a dense array, but all methods of this iterator can
// be rewritten to handle sparse token-topic matrices.
// Typical usage is as follows:
// ===============================================================
// TopicWeightIterator iter = topic_model->GetTopicWeightIterator(token);
// while (iter.NextTopic() < topic_model->topic_size()) {
//   values[iter.TopicIndex()] = iter.Weight();
// }
// ===============================================================
class TopicWeightIterator {
 public:
  // Moves the iterator to the next non-zero topic, and return the index of that topic.
  inline int NextNonZeroTopic() { return ++current_topic_; }

  // Moves the iterator to the next topic.
  inline int NextTopic() { return ++current_topic_; }

  // Returns the current position of the iterator.
  inline int TopicIndex() {return current_topic_; }

  // Returns the weight of current topic.
  // This method must not be called if TopicIndex() returns an index exceeding the number of topics.
  // It is caller responsibility to verify this condition.
  inline float Weight() {
    assert(current_topic_ < topics_count_);
    return p_w_[current_topic_];
  }

  float operator[] (int index) {
    assert(index < topics_count_);
    return p_w_[index];
  }

  // Resets the iterator to the initial state.
  inline void Reset() { current_topic_ = -1; }

 private:
  const float* p_w_;
  int topics_count_;
  mutable int current_topic_;

  TopicWeightIterator(const float* p_w, int topics_count)
      : p_w_(p_w), topics_count_(topics_count), current_topic_(-1) {}

  friend class ::artm::core::TopicModel;
  friend class ::artm::core::Regularizable;
};

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

class TokenCollectionWeights : boost::noncopyable {
 public:
  explicit TokenCollectionWeights(int topic_size) : topic_size_(topic_size) {}
  TokenCollectionWeights(int token_size, int topic_size);
  ~TokenCollectionWeights() { Clear(); }

  float get(int token_id, int topic_id) const { return values_[token_id][topic_id]; }
  void set(int token_id, int topic_id, float value) { values_[token_id][topic_id] = value; }

  const float* operator[](int token_id) const { return values_[token_id]; }
  float* operator[](int token_id) { return values_[token_id]; }

  const float* at(int token_id) const { return values_[token_id]; }
  float* at(int token_id) { return values_[token_id]; }

  size_t size() const { return values_.size(); }
  bool empty() const { return values_.empty(); }

  inline int topic_size() const { return topic_size_; }

  void Reset();
  void Clear();
  int AddToken();
  int AddToken(const Token& token, bool random_init);
  void RemoveToken(int token_id);

 private:
  int topic_size_;
  std::vector<float*> values_;
};

// A class representing a topic model.
// - ::artm::core::TopicModel is an internal representation, used in Processor, Merger,
//   and in Memcached service. It supports efficient lookup of words in the matrix.
// - ::artm::TopicModel is an external representation, implemented as protobuf message.
//   It is used to transfer model back to the user.
class TopicModel : public Regularizable {
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

  void IncreaseTokenWeight(const Token& token, int topic_id, float value);
  void IncreaseTokenWeight(int token_id, int topic_id, float value);
  void SetTokenWeight(const Token& token, int topic_id, float value);
  void SetTokenWeight(int token_id, int topic_id, float value);

  void CalcPwt() { FindPwt(&p_wt_); }
  void CalcPwt(const TokenCollectionWeights& r_wt) { FindPwt(r_wt, &p_wt_); }
  const TokenCollectionWeights& GetPwt() const { return p_wt_; }

  TopicWeightIterator GetTopicWeightIterator(const Token& token) const;
  TopicWeightIterator GetTopicWeightIterator(int token_id) const;
  const float* GetPwt(int token_id) const { return p_wt_.at(token_id); }

  ModelName model_name() const;

  int token_size() const { return token_collection_.token_size(); }
  int topic_size() const;
  google::protobuf::RepeatedPtrField<std::string> topic_name() const;

  bool has_token(const Token& token) const { return token_collection_.has_token(token); }
  int token_id(const Token& token) const { return token_collection_.token_id(token); }
  const Token& token(int index) const { return token_collection_.token(index); }

  std::map<ClassId, std::vector<float> > FindNormalizers(const TokenCollectionWeights& r_wt) const;
  std::map<ClassId, std::vector<float> > FindNormalizers() const { return FindNormalizers(TokenCollectionWeights(0)); }

  // find p_wt matrix without regularization
  virtual void FindPwt(TokenCollectionWeights* p_wt) const { return FindPwt(TokenCollectionWeights(0), p_wt); }
  // find p_wt matrix with regularization additions r_wt
  virtual void FindPwt(const TokenCollectionWeights& r_wt, TokenCollectionWeights* p_wt) const;

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
