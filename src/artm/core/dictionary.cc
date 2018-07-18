// Copyright 2017, Additive Regularization of Topic Models.

#include "artm/core/dictionary.h"
#include "artm/utility/memory_usage.h"

namespace artm {
namespace core {

void Dictionary::AddEntry(const DictionaryEntry& entry) {
  if (token_index_.find(entry.token()) != token_index_.end()) {
    LOG(WARNING) << "Token " << entry.token().keyword << " (" << entry.token().class_id
      << ") is already in dictionary";
    return;
  }

  entries_.push_back(entry);
  token_index_.insert(std::make_pair(entry.token(), entries_.size() - 1));
}

void Dictionary::AddCoocImpl(const Token& token_1, const Token& token_2, float value, CoocMap* cooc_map) {
  // check tokens are in the dictionary, e.g. exist in token_index_
  auto token_1_index = token_index_.find(token_1);
  if (token_1_index == token_index_.end()) {
    LOG(WARNING) << "No token " << token_1.keyword
                 << " (" << token_1.class_id << ") in dictionary";
    return;
  }

  auto token_2_index = token_index_.find(token_2);
  if (token_2_index == token_index_.end()) {
    LOG(WARNING) << "No token " << token_2.keyword << " (" << token_2.class_id << ") in dictionary";
    return;
  }

  AddCoocImpl(token_1_index->second, token_2_index->second, value, cooc_map);
}

void Dictionary::AddCoocImpl(int index_1, int index_2, float value, CoocMap* cooc_map) {
  // try to find token_1 in the first level of cooc_map, if no one was found, add token_1
  auto iter_1 = cooc_map->find(index_1);
  if (iter_1 == cooc_map->end()) {
    cooc_map->insert(std::make_pair(index_1, std::unordered_map<int, float>()));
    iter_1 = cooc_map->find(index_1);
  }

  // std::map::insert() ignores attempts to write several pairs with same key
  iter_1->second.insert(std::make_pair(index_2, value));
}

void Dictionary::AddCoocValue(const Token& token_1, const Token& token_2, float value) {
  AddCoocImpl(token_1, token_2, value, &cooc_values_);
}

void Dictionary::AddCoocTf(const Token& token_1, const Token& token_2, float tf) {
  AddCoocImpl(token_1, token_2, tf, &cooc_tfs_);
}

void Dictionary::AddCoocDf(const Token& token_1, const Token& token_2, float df) {
  AddCoocImpl(token_1, token_2, df, &cooc_dfs_);
}

void Dictionary::AddCoocValue(int index_1, int index_2, float value) {
  AddCoocImpl(index_1, index_2, value, &cooc_values_);
}
void Dictionary::AddCoocTf(int index_1, int index_2, float value) {
  AddCoocImpl(index_1, index_2, value, &cooc_tfs_);
}
void Dictionary::AddCoocDf(int index_1, int index_2, float value) {
  AddCoocImpl(index_1, index_2, value, &cooc_dfs_);
}

bool Dictionary::has_valid_cooc_state() const {
  if (cooc_tfs_.size() == 0 && cooc_dfs_.size() == 0) {
    return true;
  }

  return (cooc_dfs_.size() == cooc_tfs_.size()) && (cooc_dfs_.size() == cooc_values_.size());
}

int64_t Dictionary::ByteSize() const {
  int64_t retval = 0;
  retval += ::artm::utility::getMemoryUsage(entries_);
  retval += ::artm::utility::getMemoryUsage(token_index_);
  retval += ::artm::utility::getMemoryUsage(cooc_values_);
  retval += ::artm::utility::getMemoryUsage(cooc_tfs_);
  retval += ::artm::utility::getMemoryUsage(cooc_dfs_);
  for (const auto& entry : cooc_values_) {
    retval += ::artm::utility::getMemoryUsage(entry.second);
  }

  for (const auto& entry : cooc_tfs_) {
    retval += ::artm::utility::getMemoryUsage(entry.second);
  }

  for (const auto& entry : cooc_dfs_) {
    retval += ::artm::utility::getMemoryUsage(entry.second);
  }

  for (const auto& entry : entries_) {
    retval += 2 * (entry.token().keyword.size() + entry.token().class_id.size());
  }
  return retval;
}

const std::unordered_map<int, float>* Dictionary::cooc_info_impl(const Token& token, const CoocMap& cooc_map) const {
  auto index_iter = token_index_.find(token);
  if (index_iter == token_index_.end()) {
    return nullptr;
  }

  auto cooc_map_iter = cooc_map.find(index_iter->second);
  if (cooc_map_iter == cooc_map.end()) {
    return nullptr;
  }

  return &(cooc_map_iter->second);
}

const std::unordered_map<int, float>*  Dictionary::token_cooc_values(const Token& token) const {
  return cooc_info_impl(token, cooc_values_);
}

const std::unordered_map<int, float>*  Dictionary::token_cooc_tfs(const Token& token) const {
  return cooc_info_impl(token, cooc_tfs_);
}

const std::unordered_map<int, float>*  Dictionary::token_cooc_dfs(const Token& token) const {
  return cooc_info_impl(token, cooc_dfs_);
}

const DictionaryEntry* Dictionary::entry(const Token& token) const {
  auto find_iter = token_index_.find(token);
  if (find_iter != token_index_.end()) {
    return &entries_[find_iter->second];
  } else {
    return nullptr;
  }
}

const DictionaryEntry* Dictionary::entry(int index) const {
  if (index < 0 || index >= (int64_t) entries_.size()) {
    return nullptr;
  }
  return &entries_[index];
}

float Dictionary::CountTopicCoherence(const std::vector<core::Token>& tokens_to_score) {
  float coherence_value = 0.0;
  int k = static_cast<int>(tokens_to_score.size());
  if (k == 0 || k == 1) {
    return 0.0f;
  }

  // -1 means that find() result == end()
  auto indices = std::vector<int>(k, -1);
  for (int i = 0; i < k; ++i) {
    auto token_index_iter = token_index_.find(tokens_to_score[i]);
    if (token_index_iter == token_index_.end()) {
      continue;
    }
    indices[i] = token_index_iter->second;
  }

  for (int i = 0; i < k - 1; ++i) {
    if (indices[i] == -1) {
      continue;
    }

    auto cooc_map_iter = cooc_values_.find(indices[i]);
    if (cooc_map_iter == cooc_values_.end()) {
      continue;
    }

    for (int j = i; j < k; ++j) {
      if (indices[j] == -1) {
        continue;
      }

      if (tokens_to_score[j].class_id != tokens_to_score[i].class_id) {
        continue;
      }

      auto value_iter = cooc_map_iter->second.find(indices[j]);
      if (value_iter == cooc_map_iter->second.end()) {
        continue;
      }
      coherence_value += static_cast<float>(value_iter->second);
    }
  }

  return 2.0f / (k * (k - 1)) * coherence_value;
}

void Dictionary::clear() {
  name_.clear();
  entries_.clear();
  token_index_.clear();
  clear_cooc();
}

void Dictionary::clear_cooc() {
  cooc_values_.clear();
  cooc_tfs_.clear();
  cooc_dfs_.clear();
}

}  // namespace core
}  // namespace artm
// vim: set ts=2 sw=2:
