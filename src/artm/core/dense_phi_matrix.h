// Copyright 2015, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_DENSE_PHI_MATRIX_H_
#define SRC_ARTM_CORE_DENSE_PHI_MATRIX_H_

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/utility.hpp"

#include "artm/core/common.h"
#include "artm/core/phi_matrix.h"

namespace artm {
namespace core {

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

class SpinLock {
 public:
  SpinLock() : state_(kUnlocked) { }
  void Lock();
  void Unlock();

 private:
  static const bool kLocked = true;
  static const bool kUnlocked = false;
  std::atomic<bool> state_;
};

class DensePhiMatrix : boost::noncopyable, public PhiMatrix {
 public:
  explicit DensePhiMatrix(const ModelName& model_name,
                          const google::protobuf::RepeatedPtrField<std::string>& topic_name);

  virtual ~DensePhiMatrix() { Clear(); }

  virtual float get(int token_id, int topic_id) const { return values_[token_id][topic_id]; }
  virtual void set(int token_id, int topic_id, float value) { values_[token_id][topic_id] = value; }
  virtual void increase(int token_id, int topic_id, float increment) { values_[token_id][topic_id] += increment; }
  virtual void increase(int token_id, const std::vector<float>& increment);  // must be thread-safe

  virtual int topic_size() const { return topic_name_.size(); }
  virtual int token_size() const { return values_.size(); }
  virtual const Token& token(int index) const;
  virtual bool has_token(const Token& token) const;
  virtual int token_index(const Token& token) const;
  virtual google::protobuf::RepeatedPtrField<std::string> topic_name() const;
  virtual ModelName model_name() const;

  void Reset();
  int AddToken(const Token& token, bool random_init);
  void RemoveToken(const Token& token);
  void Reshape(const PhiMatrix& phi_matrix);

 private:
  void Clear();

  ModelName model_name_;
  std::vector<std::string> topic_name_;

  TokenCollection token_collection_;
  std::vector<float*> values_;
  std::vector<std::shared_ptr<SpinLock> > spin_locks_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_DENSE_PHI_MATRIX_H_
