// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <string>
#include <sstream>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/functional/hash.hpp"

#include "artm/core/common.h"

namespace artm {
namespace core {

typedef std::string ClassId;
typedef std::string TransactionTypeName;

const ClassId DefaultClass = "@default_class";
const ClassId DocumentsClass = "@documents_class";
const TransactionTypeName DefaultTransactionTypeName = "@default_transaction";

// Token is a tuple of keyword and its class_id (also known as tokens' modality).
// Pay attention to the order of the arguments in the constructor.
// For historical reasons ClassId goes first, followed by the keyword.
struct Token {
 public:
  Token(const ClassId& _class_id, const std::string& _keyword)
    : keyword(_keyword)
    , class_id(_class_id)
    , hash_(calcHash(_class_id, _keyword)) { }

  Token& operator=(const Token &token) {
    if (this != &token) {
      const_cast<std::string&>(keyword) = token.keyword;
      const_cast<ClassId&>(class_id) = token.class_id;
      const_cast<size_t&>(hash_) = token.hash_;
    }

    return *this;
  }

  bool operator<(const Token& token) const {
    if (keyword != token.keyword) {
      return keyword < token.keyword;
    }

    return class_id < token.class_id;
  }

  bool operator==(const Token& token) const {
    if (keyword == token.keyword && class_id == token.class_id) {
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

 private:
  const size_t hash_;

  static size_t calcHash(const ClassId& class_id, const std::string& keyword) {
    size_t hash = 0;
    boost::hash_combine<std::string>(hash, keyword);
    boost::hash_combine<std::string>(hash, class_id);
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
