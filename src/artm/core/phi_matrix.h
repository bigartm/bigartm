// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_PHI_MATRIX_H_
#define SRC_ARTM_CORE_PHI_MATRIX_H_


#include <string>
#include <vector>

#include "artm/messages.pb.h"

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"

namespace artm {
namespace core {

class PhiMatrix {
 public:
  static const int kUndefIndex = -1;

  virtual int token_size() const = 0;
  virtual int topic_size() const = 0;
  virtual google::protobuf::RepeatedPtrField<std::string> topic_name() const = 0;
  virtual const std::string& topic_name(int topic_id) const = 0;
  virtual ModelName model_name() const = 0;

  virtual const Token& token(int index) const = 0;
  virtual bool has_token(const Token& token) const = 0;
  virtual int token_index(const Token& token) const = 0;

  virtual float get(int token_id, int topic_id) const = 0;
  virtual void set(int token_id, int topic_id, float value) = 0;
  virtual void increase(int token_id, int topic_id, float increment) = 0;
  virtual void increase(int token_id, const std::vector<float>& increment) = 0;  // must be thread-safe

  virtual int AddToken(const Token& token) = 0;
  virtual void RemoveTokens(const std::vector<Token>& tokens) = 0;

  virtual ~PhiMatrix() {}
};


}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_PHI_MATRIX_H_
