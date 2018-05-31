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
typedef std::string TransactionTypeName;

const std::string DefaultClass = "@default_class";
const std::string DefaultTransactionTypeName = "@default_transaction";
const std::string DefaultTransactionType = "";  // includes all class_ids
const std::string DocumentsClass = "@documents_class";

// Token is a triple of keyword, its class_id (also known as tokens' modality) and typename of the transaction.
// Pay attention to the order of the arguments in the constructor.
// For historical reasons ClassId goes first, followed by the keyword and transaction typename.
struct Token {
 public:
  Token(const ClassId& _class_id, const std::string& _keyword,
        const TransactionTypeName& _transaction_typename)
    : keyword(_keyword)
    , class_id(_class_id)
    , transaction_typename(_transaction_typename)
    , hash_(calcHash(_class_id, _keyword, _transaction_typename)) { }


  Token& operator=(const Token &token) {
    if (this != &token) {
      const_cast<std::string&>(keyword) = token.keyword;
      const_cast<ClassId&>(class_id) = token.class_id;
      const_cast<TransactionTypeName&>(transaction_typename) = token.transaction_typename;
      const_cast<size_t&>(hash_) = token.hash_;
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

    return transaction_typename < token.transaction_typename;
  }

  bool operator==(const Token& token) const {
    if (keyword == token.keyword && class_id == token.class_id &&
      transaction_typename == token.transaction_typename) {
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
  const TransactionTypeName transaction_typename;

 private:
  const size_t hash_;

  static size_t calcHash(const ClassId& class_id, const std::string& keyword,
                         const TransactionTypeName& transaction_typename) {
    size_t hash = 0;
    boost::hash_combine<std::string>(hash, keyword);
    boost::hash_combine<std::string>(hash, class_id);
    boost::hash_combine<std::string>(hash, transaction_typename);
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
