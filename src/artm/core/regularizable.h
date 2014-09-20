// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_REGULARIZABLE_H_
#define SRC_ARTM_CORE_REGULARIZABLE_H_


#include <string>

#include "artm/messages.pb.h"

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"

namespace artm {
namespace core {

class TopicWeightIterator;

class Regularizable {
 public:
  virtual void IncreaseRegularizerWeight(const Token& token, int topic_id, float value) = 0;
  virtual void IncreaseRegularizerWeight(int token_id, int topic_id, float value) = 0;

  virtual int token_size() const = 0;
  virtual int topic_size() const = 0;
  virtual Token token(int index) const = 0;

  virtual TopicWeightIterator GetTopicWeightIterator(const Token& token) const = 0;
  virtual TopicWeightIterator GetTopicWeightIterator(int token_id) const = 0;

};


}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_REGULARIZABLE_H_
