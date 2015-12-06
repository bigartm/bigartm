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

class DictionaryImpl {
 public:
  explicit DictionaryImpl(const artm::DictionaryData& data);
  DictionaryImpl() { }

  void Append(const DictionaryData& dict);

  std::shared_ptr<DictionaryImpl> Duplicate() const;
    // Note: the data should be allocated, and the whole dictionary will be put
    // into it (so the method is unsafe according to 1GB limitation on proto-message size)
    // Also note, that it saves only token info, without cooc
  void StoreIntoDictionaryData(DictionaryData* data) const;

  // general method to return all cooc tokens with their values for given token
  const std::unordered_map<int, float>* cooc_info(const Token& token) const;

  const DictionaryEntryImpl* entry(const Token& token) const;
  const DictionaryEntryImpl* entry(int index) const;
  inline int size() const { return entries_.size(); }
  inline const std::unordered_map<Token, int, TokenHasher>& token_index() const { return token_index_; }
  inline const std::vector<DictionaryEntryImpl>& entries() const { return entries_; }
  inline const std::unordered_map<int, std::unordered_map<int, float> >& cooc_values() const { return cooc_values_; }

  float CountTopicCoherence(const std::vector<core::Token>& tokens_to_score);

 private:
  std::vector<DictionaryEntryImpl> entries_;
  std::unordered_map<Token, int, TokenHasher> token_index_;
  std::unordered_map<int, std::unordered_map<int, float> > cooc_values_;
};

typedef ThreadSafeCollectionHolder<std::string, DictionaryImpl> ThreadSafeDictionaryImplCollection;

class Dictionary {
 public:
  explicit Dictionary(const artm::DictionaryConfig& config);
  std::shared_ptr<Dictionary> Duplicate() const;

  inline int total_items_count() const { return total_items_count_; }
  int cooc_size(const Token& token) const;

  // cooc_value() methods return 0, if there're no such tokens
  float cooc_value(const Token& token, int index) const;
  float cooc_value(const Token& token_1, const Token& token_2) const;

  // general method to return all cooc tokens with their values for given token
  const std::unordered_map<int, float>* cooc_info(const Token& token) const;

  const DictionaryEntry* entry(const Token& token) const;
  const DictionaryEntry* entry(int index) const;
  inline int size() const { return entries_.size(); }
  inline const std::vector<DictionaryEntry>& entries() const { return entries_; }

  float CountTopicCoherence(const std::vector<core::Token>& tokens_to_score);

 private:
  int total_items_count_;
  std::vector<DictionaryEntry> entries_;
  std::unordered_map<Token, int, TokenHasher> token_index_;
  std::unordered_map<int, std::unordered_map<int, float> > cooc_values_;
};

typedef ThreadSafeCollectionHolder<std::string, Dictionary> ThreadSafeDictionaryCollection;

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_DICTIONARY_H_
