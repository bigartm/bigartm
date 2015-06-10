// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_DICTIONARY_H_
#define SRC_ARTM_CORE_DICTIONARY_H_

#include <map>
#include <string>
#include <vector>
#include <utility>

#include "artm/messages.pb.h"
#include "artm/core/common.h"
#include "artm/core/thread_safe_holder.h"

namespace artm {
namespace core {

struct TokenCoocInfo {
  TokenCoocInfo() : token(nullptr), value(0) {}
  TokenCoocInfo(const Token* _token, int _value) : token(_token), value(_value) { }
  const Token* token;
  int value;
};

class Dictionary {
 public:
  explicit Dictionary(const artm::DictionaryConfig& config);

  inline int total_items_count() const { return total_items_count_; }
  int cooc_size(const Token& token) const;
  const Token* cooc_token(const Token& token, int index) const;

  // cooc_value() methods return 0, if there're no such tokens
  int cooc_value(const Token& token, int index) const;
  int cooc_value(const Token& token_1, const Token& token_2) const;

  // general method to return all cooc tokens with their values for given token
  const std::vector<TokenCoocInfo> cooc_info(const Token& token) const;

  const DictionaryEntry* entry(const Token& token) const;
  const DictionaryEntry* entry(int index) const;
  inline int size() const { return entries_.size(); }

 private:
  int total_items_count_;
  std::vector<DictionaryEntry> entries_;
  std::unordered_map<Token, int, TokenHasher> token_index_;
  std::unordered_map<int, Token> index_token_;
  std::unordered_map<int, std::unordered_map<int, int> > cooc_values_;
};

typedef ThreadSafeCollectionHolder<std::string, Dictionary> ThreadSafeDictionaryCollection;

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_DICTIONARY_H_
