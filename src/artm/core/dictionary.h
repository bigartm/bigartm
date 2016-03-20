// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_DICTIONARY_H_
#define SRC_ARTM_CORE_DICTIONARY_H_

#include <map>
#include <string>
#include <vector>
#include <unordered_map>
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
  explicit Dictionary(const artm::DictionaryData& data);
  Dictionary() { }

  static std::vector<std::shared_ptr<artm::DictionaryData> > ImportData(const ImportDictionaryArgs& args);

  static void Export(const ExportDictionaryArgs& args,
                     ThreadSafeDictionaryCollection* dictionaries);

  static std::pair<std::shared_ptr<DictionaryData>, std::shared_ptr<DictionaryData> >
    Gather(const GatherDictionaryArgs& args, const ThreadSafeCollectionHolder<std::string, Batch>& mem_batches);

  static std::pair<std::shared_ptr<DictionaryData>, std::shared_ptr<DictionaryData> >
    Filter(const FilterDictionaryArgs& args, ThreadSafeDictionaryCollection* dictionaries);

  void Append(const DictionaryData& dict);

  std::shared_ptr<Dictionary> Duplicate() const;
    // Note: the data should be allocated, and the whole dictionary will be put
    // into it (so the method is unsafe according to 1GB limitation on proto-message size)
    // Also note, that it saves only token info, without cooc
  void StoreIntoDictionaryData(DictionaryData* data) const;

  // general method to return all cooc tokens with their values for given token
  const std::unordered_map<int, float>* cooc_info(const Token& token) const;

  const DictionaryEntry* entry(const Token& token) const;
  const DictionaryEntry* entry(int index) const;
  inline size_t size() const { return entries_.size(); }
  inline const std::unordered_map<Token, int, TokenHasher>& token_index() const { return token_index_; }
  inline const std::vector<DictionaryEntry>& entries() const { return entries_; }
  inline const std::unordered_map<int, std::unordered_map<int, float> >& cooc_values() const { return cooc_values_; }

  float CountTopicCoherence(const std::vector<core::Token>& tokens_to_score);

 private:
  std::vector<DictionaryEntry> entries_;
  std::unordered_map<Token, int, TokenHasher> token_index_;
  std::unordered_map<int, std::unordered_map<int, float> > cooc_values_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_DICTIONARY_H_
