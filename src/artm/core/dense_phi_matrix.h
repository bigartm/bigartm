// Copyright 2015, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_DENSE_PHI_MATRIX_H_
#define SRC_ARTM_CORE_DENSE_PHI_MATRIX_H_

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "boost/utility.hpp"

#include "artm/core/common.h"
#include "artm/core/phi_matrix.h"

namespace artm {
namespace core {

class TokenCollection {
 public:
  void Clear();
  int  AddToken(const Token& token);
  void Swap(TokenCollection* rhs);

  int token_size() const;
  bool has_token(const Token& token) const;
  int token_id(const Token& token) const;
  const Token& token(int index) const;

 private:
  std::unordered_map<Token, int, TokenHasher> token_to_token_id_;
  std::vector<Token> token_id_to_token_;
};

class SpinLock : boost::noncopyable {
 public:
  SpinLock() : state_(kUnlocked) { }
  void Lock();
  void Unlock();

 private:
  static const bool kLocked = true;
  static const bool kUnlocked = false;
  std::atomic<bool> state_;
};

class PhiMatrixFrame : public PhiMatrix {
 public:
  explicit PhiMatrixFrame(const ModelName& model_name,
                          const google::protobuf::RepeatedPtrField<std::string>& topic_name);

  virtual ~PhiMatrixFrame() { }

  virtual int topic_size() const { return topic_name_.size(); }
  virtual int token_size() const { return token_collection_.token_size(); }
  virtual const Token& token(int index) const;
  virtual bool has_token(const Token& token) const;
  virtual int token_index(const Token& token) const;
  virtual google::protobuf::RepeatedPtrField<std::string> topic_name() const;
  virtual const std::string& topic_name(int topic_id) const;
  virtual ModelName model_name() const;

  void Clear();
  virtual int AddToken(const Token& token);

  void Lock(int token_id) { spin_locks_[token_id]->Lock(); }
  void Unlock(int token_id) { spin_locks_[token_id]->Unlock(); }

  void Swap(PhiMatrixFrame* rhs);

  PhiMatrixFrame(const PhiMatrixFrame& rhs);
  PhiMatrixFrame& operator=(const PhiMatrixFrame&);

 private:
  ModelName model_name_;
  std::vector<std::string> topic_name_;

  TokenCollection token_collection_;
  std::vector<std::shared_ptr<SpinLock> > spin_locks_;
};

class DensePhiMatrix;
class AttachedPhiMatrix;

class DensePhiMatrix : public PhiMatrixFrame {
 public:
  explicit DensePhiMatrix(const ModelName& model_name,
                          const google::protobuf::RepeatedPtrField<std::string>& topic_name);

  virtual ~DensePhiMatrix() { Clear(); }

  virtual std::shared_ptr<PhiMatrix> Duplicate() const;

  virtual float get(int token_id, int topic_id) const { return values_[token_id][topic_id]; }
  virtual void set(int token_id, int topic_id, float value) { values_[token_id][topic_id] = value; }
  virtual void increase(int token_id, int topic_id, float increment) { values_[token_id][topic_id] += increment; }
  virtual void increase(int token_id, const std::vector<float>& increment);  // must be thread-safe

  virtual void Clear();
  virtual int AddToken(const Token& token);
  virtual void RemoveTokens(const std::vector<Token>& tokens);

  void Reset();
  void Reshape(const PhiMatrix& phi_matrix);

 private:
  friend class AttachedPhiMatrix;
  DensePhiMatrix(const DensePhiMatrix& rhs);
  explicit DensePhiMatrix(const AttachedPhiMatrix& rhs);
  DensePhiMatrix& operator=(const PhiMatrixFrame&);

  std::vector<float*> values_;
};

class AttachedPhiMatrix : boost::noncopyable, public PhiMatrixFrame {
 public:
  AttachedPhiMatrix(int address_length, float* address, PhiMatrixFrame* source);
  virtual ~AttachedPhiMatrix() { values_.clear(); }  // DO NOT delete this memory; AttachedPhiMatrix do not own it.

  virtual std::shared_ptr<PhiMatrix> Duplicate() const;

  virtual float get(int token_id, int topic_id) const { return values_[token_id][topic_id]; }
  virtual void set(int token_id, int topic_id, float value) { values_[token_id][topic_id] = value; }
  virtual void increase(int token_id, int topic_id, float increment) { values_[token_id][topic_id] += increment; }
  virtual void increase(int token_id, const std::vector<float>& increment);  // must be thread-safe

  virtual void Clear();
  virtual int AddToken(const Token& token);
  virtual void RemoveTokens(const std::vector<Token>& tokens);

 private:
  friend class DensePhiMatrix;
  std::vector<float*> values_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_DENSE_PHI_MATRIX_H_
