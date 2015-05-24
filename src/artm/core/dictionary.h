// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_DICTIONARY_H_
#define SRC_ARTM_CORE_DICTIONARY_H_

#include <map>
#include <string>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/core/common.h"
#include "artm/core/thread_safe_holder.h"

namespace artm {
namespace core {

class Dictionary {
 public:
  explicit Dictionary(const artm::DictionaryConfig& config);

  int cooc_size(const Token& token) const;
  const Token* cooc_token(const Token& token, int index) const;

  // cooc_value() methods return 0, if there're no such tokens
  int cooc_value(const Token& token, int index) const;
  int cooc_value(const Token& token_1, const Token& token_2) const;

  const DictionaryEntry* entry(const Token& token) const;
  const DictionaryEntry* entry(int index) const { return &entries_[index]; }
  inline int size() const { return entries_.size(); }

 private:
  std::vector<DictionaryEntry> entries_;
  std::map<Token, int> token_index_;
  std::map<int, Token> index_token_;
  std::map<int, std::map<int, int> > cooc_values_;
};

typedef ThreadSafeCollectionHolder<std::string, Dictionary> ThreadSafeDictionaryCollection;

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_DICTIONARY_H_
