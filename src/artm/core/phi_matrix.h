// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_REGULARIZABLE_H_
#define SRC_ARTM_CORE_REGULARIZABLE_H_


#include <string>
#include <vector>

#include "artm/messages.pb.h"

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"

namespace artm {
namespace core {

class TopicWeightIterator;
class TokenCollectionWeights;

class PhiMatrix {
 public:
  virtual int token_size() const = 0;
  virtual int topic_size() const = 0;
  virtual google::protobuf::RepeatedPtrField<std::string> topic_name() const = 0;
  virtual const Token& token(int index) const = 0;

  virtual float get(int token_id, int topic_id) const = 0;
  virtual void set(int token_id, int topic_id, float value) = 0;

  virtual ~PhiMatrix() {}
};


}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_REGULARIZABLE_H_
