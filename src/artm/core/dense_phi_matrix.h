// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

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

// TokenCollection class represents a sequential vector of tokens.
// It also contains a mapping from Token to its index for efficient lookup.
// For tokens that are not present in the collection loop up method will return 'UnknownId' constant.
class TokenCollection {
 public:
  void Clear();
  int  AddToken(const Token& token);
  void Swap(TokenCollection* rhs);
  int64_t ByteSize() const;

  int token_size() const;
  bool has_token(const Token& token) const;
  int token_id(const Token& token) const;
  const Token& token(int index) const;

 private:
  std::unordered_map<Token, int, TokenHasher> token_to_token_id_;
  std::vector<Token> token_id_to_token_;
};

// A simple spin lock class, used for synchronization.
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

// PhiMatrixFrame is a abstract class that partially implements PhiMatrix interface.
// It implements most methods that manage the structure of the PhiMatrix
// (e.g. the set of tokens, and the set of topic names).
// It does not implement the actual storate for the 2D matrix (e.g. n_wt or p(w|t) values).
// This storate is implemented in derived classes DensePhiMatrix and AttachedPhiMatrix.
class PhiMatrixFrame : public PhiMatrix {
 public:
  explicit PhiMatrixFrame(const ModelName& model_name,
                          const google::protobuf::RepeatedPtrField<std::string>& topic_name);

  virtual ~PhiMatrixFrame() { }

  virtual int topic_size() const { return static_cast<int>(topic_name_.size()); }
  virtual int token_size() const { return static_cast<int>(token_collection_.token_size()); }
  virtual const Token& token(int index) const;
  virtual bool has_token(const Token& token) const;
  virtual int token_index(const Token& token) const;
  virtual google::protobuf::RepeatedPtrField<std::string> topic_name() const;
  virtual const std::string& topic_name(int topic_id) const;
  virtual void set_topic_name(int topic_id, const std::string& topic_name);
  virtual ModelName model_name() const;
  virtual int64_t ByteSize() const;

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

// PackedValues class represents one row of Phi matrix.
// Sparse rows (with many zeros) might be packed for memory efficiency.
class PackedValues {
 public:
  PackedValues();
  explicit PackedValues(int size);
  explicit PackedValues(const PackedValues& rhs);
  PackedValues(const float* values, int size);
  virtual int64_t ByteSize() const;

  bool is_packed() const;
  float get(int index) const;
  void get(std::vector<float>* buffer) const;
  float* unpack();
  void pack();
  void reset(int size);

 private:
  std::vector<float> values_;
  std::vector<bool> bitmask_;
  std::vector<int> ptr_;
};

// DensePhiMatrix class implements PhiMatrix interface as a dense matrix.
// The class owns the memory allocated to store the elements.
class DensePhiMatrix : public PhiMatrixFrame {
 public:
  explicit DensePhiMatrix(const ModelName& model_name,
                          const google::protobuf::RepeatedPtrField<std::string>& topic_name);

  virtual ~DensePhiMatrix() { Clear(); }
  virtual int64_t ByteSize() const;

  virtual std::shared_ptr<PhiMatrix> Duplicate() const;

  virtual float get(int token_id, int topic_id) const;
  virtual void get(int token_id, std::vector<float>* buffer) const;
  virtual void set(int token_id, int topic_id, float value);
  virtual void increase(int token_id, int topic_id, float increment);
  virtual void increase(int token_id, const std::vector<float>& increment);  // must be thread-safe

  virtual void Clear();
  virtual int AddToken(const Token& token);

  void Reset();
  void Reshape(const PhiMatrix& phi_matrix);

 private:
  friend class AttachedPhiMatrix;
  DensePhiMatrix(const DensePhiMatrix& rhs);
  explicit DensePhiMatrix(const AttachedPhiMatrix& rhs);
  DensePhiMatrix& operator=(const PhiMatrixFrame&);

  std::vector<PackedValues> values_;
};

// DensePhiMatrix class implements PhiMatrix interface as a dense matrix.
// The class DOES NOT own the memory, allocated to store the elements.
// Instead the memory is provided by external code.
// Typically it will be stored in a numpy matrix in the Python interface.
class AttachedPhiMatrix : boost::noncopyable, public PhiMatrixFrame {
 public:
  AttachedPhiMatrix(int address_length, float* address, PhiMatrixFrame* source);
  virtual ~AttachedPhiMatrix() { values_.clear(); }  // DO NOT delete this memory; AttachedPhiMatrix do not own it.
  virtual int64_t ByteSize() const { return 0; }

  virtual std::shared_ptr<PhiMatrix> Duplicate() const;

  virtual float get(int token_id, int topic_id) const { return values_[token_id][topic_id]; }
  virtual void get(int token_id, std::vector<float>* buffer) const;
  virtual void set(int token_id, int topic_id, float value) { values_[token_id][topic_id] = value; }
  virtual void increase(int token_id, int topic_id, float increment) { values_[token_id][topic_id] += increment; }
  virtual void increase(int token_id, const std::vector<float>& increment);  // must be thread-safe

  virtual void Clear();
  virtual int AddToken(const Token& token);

 private:
  friend class DensePhiMatrix;
  std::vector<float*> values_;
};

}  // namespace core
}  // namespace artm
