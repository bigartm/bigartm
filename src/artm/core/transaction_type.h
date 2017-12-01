// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <string>
#include <sstream>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/functional/hash.hpp"

#include "artm/core/common.h"
#include "artm/core/transaction_type.h"

namespace artm {
namespace core {

class TransactionType {
 public:
  TransactionType() : data_(std::string()), hash_(calcHash(std::string())) { }
  explicit TransactionType(const std::string& src) : data_(src), hash_(calcHash(src)) { }

  explicit TransactionType(const std::vector<std::string>& src) {
    data_.clear();
    std::stringstream ss;
    for (int i = 0; i < src.size(); ++i) {
      if (i > 0) {
        ss << TransactionSeparator;
      }
      ss << src[i];
    }
    data_ = ss.str();
    hash_ = calcHash(data_);
  }

  TransactionType& operator=(const TransactionType &rhs) {
    if (this != &rhs) {
      const_cast<std::string&>(data_) = rhs.data_;
      const_cast<size_t&>(hash_) = rhs.hash_;
    }

    return *this;
  }

  const std::string& AsString() const { return data_; }

  std::vector<std::string> AsVector() const {
    return TransactionTypeStrAsVector(this->data_);
  }

  static std::vector<std::string> TransactionTypeStrAsVector(const std::string& tt) {
    std::vector<std::string> retval;
    boost::split(retval, tt, boost::is_any_of(TransactionSeparator));
    return retval;
  }

  bool ContainsIn(const google::protobuf::RepeatedPtrField<std::string>& tts) const {
    for (const auto& t : tts) {
      if (t == this->AsString()) {
        return true;
      }
    }
    return false;
  }

  bool operator<(const TransactionType& tt) const {
    return AsString() < tt.AsString();
  }

  bool operator==(const TransactionType& tt) const {
    return AsString() == tt.AsString();
  }

  bool operator!=(const TransactionType& tt) const {
    return !(*this == tt);
  }

  size_t hash() const { return hash_; }

 private:
  std::string data_;
  size_t hash_;

  static size_t calcHash(const std::string& data) {
    size_t hash = 0;
    boost::hash_combine<std::string>(hash, data);
    return hash;
  }
};

struct TransactionHasher {
  size_t operator()(const TransactionType& tt) const {
    return tt.hash();
  }
};

}  // namespace core
}  // namespace artm
