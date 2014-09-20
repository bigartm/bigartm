// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_TOPIC_MODEL_H_
#define SRC_ARTM_CORE_TOPIC_MODEL_H_

#include <assert.h>

#include <algorithm>
#include <map>
#include <vector>
#include <set>
#include <string>

#include "boost/utility.hpp"
#include "boost/uuid/uuid.hpp"

#include "artm/messages.pb.h"

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"
#include "artm/core/regularizable.h"

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
    return std::max<float>(n_w_[current_topic_] + r_w_[current_topic_], 0.0f) / n_t_[current_topic_];
  }

  // Not normalized weight.
  inline float NotNormalizedWeight() {
    assert(current_topic_ < topics_count_);
    return n_w_[current_topic_];
  }

  inline float NotNormalizedRegularizerWeight() {
    assert(current_topic_ < topics_count_);
    return r_w_[current_topic_];
  }

  inline const float* GetNormalizer() { return n_t_; }
  inline const float* GetRegularizer() { return r_w_; }
  inline const float* GetData() { return n_w_; }

  // Resets the iterator to the initial state.
  inline void Reset() { current_topic_ = -1; }

 private:
  const float* n_w_;
  const float* r_w_;
  const float* n_t_;
  int topics_count_;
  mutable int current_topic_;

  TopicWeightIterator(const float* n_w,
                      const float* r_w,
                      const float* n_t,
                      int topics_count)
      : n_w_(n_w),
        r_w_(r_w),
        n_t_(n_t),
        topics_count_(topics_count),
        current_topic_(-1) {
    assert(n_w != nullptr);
    assert(r_w != nullptr);
    assert(n_t != nullptr);
  }

  friend class ::artm::core::TopicModel;
  friend class ::artm::core::Regularizable;
};

// A class representing a topic model.
// - ::artm::core::TopicModel is an internal representation, used in Processor, Merger,
//   and in Memcached service. It supports efficient lookup of words in the matrix.
// - ::artm::TopicModel is an external representation, implemented as protobuf message.
//   It is used to transfer model back to user, or between Merger and MemcachedService.
// - ::artm::core::ModelIncrement is very similar to the model, but it used to represent
//   an increment to the model. This representation is used to transfer an updates from
//   processor to Merger, and from Merger to MemcachedService.
class TopicModel : public Regularizable {
 public:
  explicit TopicModel(ModelName model_name, int topics_count);
  explicit TopicModel(const TopicModel& rhs);
  explicit TopicModel(const ::artm::TopicModel& external_topic_model);
  explicit TopicModel(const ::artm::core::ModelIncrement& model_increment);

  void Clear(ModelName model_name, int topics_count);
  ~TopicModel();

  void RetrieveExternalTopicModel(::artm::TopicModel* topic_model) const;
  void CopyFromExternalTopicModel(const ::artm::TopicModel& topic_model);

  void RetrieveModelIncrement(::artm::core::ModelIncrement* diff) const;

  // Applies model increment to this TopicModel.
  void ApplyDiff(const ::artm::core::ModelIncrement& diff);
  void ApplyDiff(const ::artm::core::TopicModel& diff);

  int  AddToken(const Token& token, bool random_init = true);
  int  AddToken(ClassId class_id, std::string keyword, bool random_init);
  void IncreaseTokenWeight(const Token& token, int topic_id, float value);
  void IncreaseTokenWeight(int token_id, int topic_id, float value);
  void SetTokenWeight(const Token& token, int topic_id, float value);
  void SetTokenWeight(int token_id, int topic_id, float value);

  void IncreaseRegularizerWeight(const Token& token, int topic_id, float value);
  void IncreaseRegularizerWeight(int token_id, int topic_id, float value);
  void SetRegularizerWeight(const Token& token, int topic_id, float value);
  void SetRegularizerWeight(int token_id, int topic_id, float value);

  TopicWeightIterator GetTopicWeightIterator(const Token& token) const;
  TopicWeightIterator GetTopicWeightIterator(int token_id) const;

  ModelName model_name() const;

  int token_size() const;
  int topic_size() const;

  bool has_token(const Token& token) const;
  int token_id(const Token& token) const;
  Token token(int index) const;

 private:
  ModelName model_name_;

  std::map<Token, int> token_to_token_id_;
  std::vector<Token> token_id_to_token_;
  int topics_count_;

  std::vector<float*> n_wt_;  // vector of length tokens_count
  std::vector<float*> r_wt_;  // regularizer's additions
  // normalization constant for each topic in each Phi
  std::map<ClassId, std::vector<float> > n_t_;
  // pointer to the vector of default_class
  std::vector<float>* n_t_default_class_;

  std::vector<boost::uuids::uuid> batch_uuid_;  // batches contributing to this model

  void CreateNormalizerVector(ClassId class_id, int no_topics);
  const std::vector<float>* GetNormalizerVector(const ClassId& class_id) const;
  std::vector<float>* GetNormalizerVector(const ClassId& class_id);
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_TOPIC_MODEL_H_
