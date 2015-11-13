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

class DictionaryEntryImpl {
 public: 
  DictionaryEntryImpl(Token token, float value, float tf, float df)
    : token_(token), token_value_(value), token_tf_(tf), token_df_(df) { }
  
  const Token& token() const { return token_; }
  float token_value() const { return token_value_; }
  float token_tf() const { return token_tf_; }
  float token_df() const { return token_df_; }

 private:
  Token token_;
  float token_value_;
  float token_tf_;
  float token_df_;
};

class Dictionary {
 public:
  explicit Dictionary(const artm::DictionaryData& data);
  Dictionary() { }

  void Append (const DictionaryData& dict);

  std::shared_ptr<Dictionary> Duplicate() const;
    // Note: the data should be allocated, and the whole dictionary will be put
    // into it (so the method is unsafe according to 1GB limitation on proto-message size)
    // Also note, that it saves only token info, without cooc
  void StoreIntoDictionaryData(DictionaryData* data) const;

  // general method to return all cooc tokens with their values for given token
  const std::unordered_map<int, float>* cooc_info(const Token& token) const;

  const DictionaryEntryImpl* entry(const Token& token) const;
  const DictionaryEntryImpl* entry(int index) const;
  inline int size() const { return entries_.size(); }
  inline const std::unordered_map<int, Token>& index_token() const { return index_token_; }

  float CountTopicCoherence(const std::vector<core::Token>& tokens_to_score);

 private:
  std::vector<DictionaryEntryImpl> entries_;
  std::unordered_map<Token, int, TokenHasher> token_index_;
  std::unordered_map<int, Token> index_token_;
  std::unordered_map<int, std::unordered_map<int, float> > cooc_values_;
};

typedef ThreadSafeCollectionHolder<std::string, Dictionary> ThreadSafeDictionaryCollection;

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_DICTIONARY_H_
