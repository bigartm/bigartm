// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <unordered_set>

#include "boost/algorithm/string.hpp"
#include "boost/functional/hash.hpp"

#include "artm/core/common.h"
#include "artm/core/token.h"

namespace artm {
namespace core {

class TransactionType {
 public:
  TransactionType()
    : str_data_(std::string())
    , set_data_(std::unordered_set<ClassId>())
    , hash_(calcHash(std::string())) { }

  explicit TransactionType(const std::string& src)
    : str_data_(src)
    , set_data_(TransactionTypeAsSet(src))
    , hash_(calcHash(src)) { }

  explicit TransactionType(const std::unordered_set<ClassId>& src)
    : str_data_(TransactionTypeAsStr(src))
    , set_data_(src)
    , hash_(-1) {
    hash_ = calcHash(str_data_);
  }

  const std::string& AsString() const { return str_data_; }
  const std::unordered_set<ClassId>& AsSet() const { return set_data_; }

  size_t hash() const { return hash_; }

  TransactionType& operator=(const TransactionType &rhs) {
    if (this != &rhs) {
      const_cast<std::string&>(str_data_) = rhs.str_data_;
      const_cast<size_t&>(hash_) = rhs.hash_;
    }

    return *this;
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

 private:
  std::string str_data_;
  std::unordered_set<ClassId> set_data_;
  size_t hash_;

  std::string TransactionTypeAsStr(const std::unordered_set<ClassId>& src) {
    std::stringstream ss;
    int i = 0;
    for (const auto& e : src) {
      ss << ((i++ > 0) ? "^" : "") << e;
    }
    return ss.str();
  }

  std::unordered_set<std::string> TransactionTypeAsSet(const std::string& tt) {
    std::unordered_set<ClassId> retval;
    boost::split(retval, tt, boost::is_any_of("^"));
    return retval;
  }

  size_t calcHash(const std::string& data) {
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
