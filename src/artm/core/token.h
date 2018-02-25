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

typedef std::string ClassId;
const std::string DefaultClass = "@default_class";
const std::string DocumentsClass = "@documents_class";

// Token is a triple of keyword, its class_id (also known as tokens' modality) and type of the transaction.
// Pay attention to the order of the arguments in the constructor.
// For historical reasons ClassId goes first, followed by the keyword and transaction type.
struct Token {
 public:
  Token(const ClassId& _class_id, const std::string& _keyword)
      : keyword(_keyword), class_id(_class_id), transaction_type(_class_id)
      , hash_(calcHash(_class_id, _keyword, TransactionType(_class_id))) { }

  Token(const ClassId& _class_id, const std::string& _keyword,
        const TransactionType& transaction_type, bool check = false)
    : keyword(_keyword), class_id(_class_id), transaction_type(transaction_type)
    , hash_(calcHash(_class_id, _keyword, transaction_type)) {
    if (check) {
      for (const auto& elem : transaction_type.AsVector()) {
        if (class_id == elem) {
          return;
        }
      }
      LOG(ERROR) << "Transaction type ( " << transaction_type.AsString() << " ) of token ( " << _keyword
        << " ) does not contain token's class_id ( " << _class_id << " )";
    }
  }


  Token& operator=(const Token &rhs) {
    if (this != &rhs) {
      const_cast<std::string&>(keyword) = rhs.keyword;
      const_cast<ClassId&>(class_id) = rhs.class_id;
      const_cast<TransactionType&>(transaction_type) = rhs.transaction_type;
      const_cast<size_t&>(hash_) = rhs.hash_;
    }

    return *this;
  }

  bool operator<(const Token& token) const {
    if (keyword != token.keyword) {
      return keyword < token.keyword;
    }

    if (class_id != token.class_id) {
      return class_id < token.class_id;
    }

    return transaction_type.AsString() < token.transaction_type.AsString();
  }

  bool operator==(const Token& token) const {
    if (keyword == token.keyword && class_id == token.class_id &&
        transaction_type.AsString() == token.transaction_type.AsString()) {
      return true;
    }
    return false;
  }

  bool operator!=(const Token& token) const {
    return !(*this == token);
  }

  size_t hash() const { return hash_; }

  const std::string keyword;
  const ClassId class_id;
  const TransactionType transaction_type;

 private:
  const size_t hash_;

  static size_t calcHash(const ClassId& class_id, const std::string& keyword,
                         const TransactionType& transaction_type) {
    size_t hash = 0;
    boost::hash_combine<std::string>(hash, keyword);
    boost::hash_combine<std::string>(hash, class_id);
    boost::hash_combine<std::string>(hash, transaction_type.AsString());
    return hash;
  }
};

struct TokenHasher {
  size_t operator()(const Token& token) const {
    return token.hash();
  }
};

}  // namespace core
}  // namespace artm
