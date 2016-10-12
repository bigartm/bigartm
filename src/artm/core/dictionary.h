// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_DICTIONARY_H_
#define SRC_ARTM_CORE_DICTIONARY_H_

#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

#include "boost/serialization/serialization.hpp"
#include "boost/serialization/version.hpp"
#include "boost/serialization/vector.hpp"

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
  // Default constructor - used for serialization only
  DictionaryEntry() : token_(), token_value_(0.0f), token_tf_(0.0f), token_df_(0.0f) {}

  DictionaryEntry(Token token, float value, float tf, float df)
    : token_(token), token_value_(value), token_tf_(tf), token_df_(df) { }

  bool operator==(const DictionaryEntry& rhs) const;
  bool operator!=(const DictionaryEntry& rhs) const;

  const Token& token() const { return token_; }
  float token_value() const { return token_value_; }
  float token_tf() const { return token_tf_; }
  float token_df() const { return token_df_; }

  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {  // NOLINT
    ar & token_;
    ar & token_value_;
    ar & token_tf_;
    ar & token_df_;
  }

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

  bool operator==(const Dictionary& rhs) const;
  bool operator!=(const Dictionary& rhs) const;

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

  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {  // NOLINT
    ar & entries_;
    ar & num_items_in_collection_;

    // ar & cooc_values_; //
    if (Archive::is_loading::value) {
      size_t cooc_size;
      ar & cooc_size;
      for (size_t cooc_item = 0; cooc_item < cooc_size; cooc_item++) {
        int left, right;
        float value;
        ar & left;
        ar & right;
        ar & value;
        cooc_values_[left][right] = value;
      }
    } else {
      size_t cooc_size = 0;
      for (auto& iter1 : cooc_values_) cooc_size += iter1.second.size();
      ar & cooc_size;
      for (auto& iter1 : cooc_values_) {
        for (auto& iter2 : iter1.second) {
          int left = iter1.first;
          int right = iter2.first;
          float value = iter2.second;
          ar & left;
          ar & right;
          ar & value;
        }
      }
    }
    // ar & cooc_values_; //

    if (Archive::is_loading::value) {
      for (int token_index = 0; token_index < entries_.size(); ++token_index)
        token_index_.insert(std::make_pair(entries_[token_index].token(), token_index));
    }
  }

 private:
  std::vector<DictionaryEntry> entries_;
  std::unordered_map<Token, int, TokenHasher> token_index_;
  std::unordered_map<int, std::unordered_map<int, float> > cooc_values_;
  size_t num_items_in_collection_;
};

}  // namespace core
}  // namespace artm

BOOST_CLASS_VERSION(::artm::core::DictionaryEntry, 0)  // NOLINT
BOOST_CLASS_VERSION(::artm::core::Dictionary, 0)  // NOLINT

#endif  // SRC_ARTM_CORE_DICTIONARY_H_
