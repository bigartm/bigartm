// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "artm/core/common.h"
#include "artm/core/thread_safe_holder.h"
#include "artm/core/token.h"

namespace artm {
namespace core {

class Dictionary;

// ThreadSafeDictionaryCollection is a collection of dictionaries.
// It is typically accessed via a ThreadSafeDictionaryCollection::singleton(),
// which ensures that all master components share the same set of dictionaries.
// The key (std::string) corresponds to the name of the dictionary.
typedef ThreadSafeCollectionHolder<std::string, Dictionary> ThreadSafeDictionaryCollection;

typedef std::unordered_map<int, std::unordered_map<int, float> > CoocMap;

// DictionaryEntry represents one entry in the dictionary, associated with a specific token.
class DictionaryEntry {
 public:
  DictionaryEntry(Token token, float value, float tf, float df)
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

// Dictionary is a sequential vector of dictionary entries.
// It is important that the entries in the dictionary can be accessed by their index.
// For example, if a dictionary is used to initialize the PhiMatrix the order of the
// entries will define the order of tokens in the PhiMatrix.
// Dictionary also supports an efficient lookup of the entries by its token.
// Dictionary also stores a co-occurence data, used by Coherence score and regularizer.
class Dictionary {
 public:
  explicit Dictionary(const std::string& name) : name_(name) { }

  // SECTION OF SETTERS
  void AddEntry(const DictionaryEntry& entry);

  void AddCoocValue(const Token& token_1, const Token& token_2, float value);
  void AddCoocTf(const Token& token_1, const Token& token_2, float value);
  void AddCoocDf(const Token& token_1, const Token& token_2, float value);

  void AddCoocValue(int index_1, int index_2, float value);
  void AddCoocTf(int index_1, int index_2, float value);
  void AddCoocDf(int index_1, int index_2, float value);

  void SetNumItems(int num_items) { num_items_in_collection_ = num_items; }

  // SECTION OF GETTERS
  bool HasToken(const Token& token) const { return token_index_.find(token) != token_index_.end(); }

  // general method to return all cooc tokens with their values for given token
  const std::unordered_map<int, float>* token_cooc_values(const Token& token) const;
  const std::unordered_map<int, float>* token_cooc_tfs(const Token& token) const;
  const std::unordered_map<int, float>* token_cooc_dfs(const Token& token) const;

  const DictionaryEntry* entry(const Token& token) const;
  const DictionaryEntry* entry(int index) const;

  int size() const { return entries_.size(); }
  int num_items() const { return num_items_in_collection_; }
  const std::string& name() const { return name_; }
  bool has_valid_cooc_state() const;
  int64_t ByteSize() const;

  const std::vector<DictionaryEntry>& entries() const { return entries_; }
  const std::unordered_map<Token, int, TokenHasher>& token_index() const { return token_index_; }

  const std::unordered_map<int, std::unordered_map<int, float> >& cooc_values() const { return cooc_values_; }
  const std::unordered_map<int, std::unordered_map<int, float> >& cooc_tfs() const { return cooc_tfs_; }
  const std::unordered_map<int, std::unordered_map<int, float> >& cooc_dfs() const { return cooc_dfs_; }

  // SECTION OF OPERATIONS
  float CountTopicCoherence(const std::vector<core::Token>& tokens_to_score);

  std::shared_ptr<Dictionary> Duplicate() const;

  void clear();
  void clear_cooc();

 private:
  std::string name_;
  std::vector<DictionaryEntry> entries_;
  std::unordered_map<Token, int, TokenHasher> token_index_;
  CoocMap cooc_values_;
  CoocMap cooc_tfs_;
  CoocMap cooc_dfs_;
  size_t num_items_in_collection_;

  void AddCoocImpl(const Token& token_1, const Token& token_2, float value, CoocMap* cooc_map);
  void AddCoocImpl(int index_1, int index_2, float value, CoocMap* cooc_map);
  const std::unordered_map<int, float>* cooc_info_impl(const Token& token, const CoocMap& cooc_map) const;
};

}  // namespace core
}  // namespace artm
