// Copyright 2014, Additive Regularization of Topic Models.

#include <string>
#include <map>
#include <memory>

#include "artm/core/dictionary.h"

namespace artm {
namespace core {

Dictionary::Dictionary(const artm::DictionaryConfig& config) {
  total_items_count_ = config.total_items_count();
  for (int index = 0; index < config.entry_size(); ++index) {
    const ::artm::DictionaryEntry& entry = config.entry(index);
    ClassId class_id;
    if (entry.has_class_id()) class_id = entry.class_id();
    else                      class_id = DefaultClass;
    token_index_.insert(std::make_pair(Token(class_id, entry.key_token()), index));
    index_token_.insert(std::make_pair(index, Token(class_id, entry.key_token())));
    entries_.push_back(entry);
  }

  if (config.has_cooc_entries()) {
    for (int i = 0; i < config.cooc_entries().first_index_size(); ++i) {
      auto& first_entry = config.entry(config.cooc_entries().first_index(i));
      auto& second_entry = config.entry(config.cooc_entries().second_index(i));
      auto first_index_iter = token_index_.find(Token(first_entry.class_id(), first_entry.key_token()));
      auto second_index_iter = token_index_.find(Token(second_entry.class_id(), second_entry.key_token()));

      // ignore tokens, that are not represented in dictionary entries
      if (first_index_iter != token_index_.end() && second_index_iter != token_index_.end()) {
        auto first_cooc_iter = cooc_values_.find(first_index_iter->second);
        if (first_cooc_iter == cooc_values_.end()) {
          cooc_values_.insert(std::make_pair(first_index_iter->second, std::unordered_map<int, float>()));
          first_cooc_iter = cooc_values_.find(first_index_iter->second);
        }

        // std::map::insert() ignores attempts to write several pairs with same key
        first_cooc_iter->second.insert(std::make_pair(second_index_iter->second,
                                                      config.cooc_entries().value(i)));

        if (config.cooc_entries().symmetric_cooc_values()) {
          auto second_cooc_iter = cooc_values_.find(second_index_iter->second);
          if (second_cooc_iter == cooc_values_.end()) {
            cooc_values_.insert(std::make_pair(second_index_iter->second, std::unordered_map<int, float>()));
            second_cooc_iter = cooc_values_.find(second_index_iter->second);
          }

          second_cooc_iter->second.insert(std::make_pair(first_index_iter->second,
                                                         config.cooc_entries().value(i)));
        }
      }
    }
  }
}

std::shared_ptr<Dictionary> Dictionary::Duplicate() const {
  return std::shared_ptr<Dictionary>(new Dictionary(*this));
}

int Dictionary::cooc_size(const Token& token) const {
  auto index_iter = token_index_.find(token);
  if (index_iter == token_index_.end()) return 0;

  auto cooc_map_iter = cooc_values_.find(index_iter->second);
  if (cooc_map_iter == cooc_values_.end()) return 0;

  return cooc_map_iter->second.size();
}

const Token* Dictionary::cooc_token(const Token& token, int index) const {
  auto index_iter = token_index_.find(token);
  if (index_iter == token_index_.end()) return nullptr;

  auto cooc_map_iter = cooc_values_.find(index_iter->second);
  if (cooc_map_iter == cooc_values_.end()) return nullptr;

  int internal_index = -1;
  for (auto iter = cooc_map_iter->second.begin(); iter != cooc_map_iter->second.end(); ++iter)
    if (++internal_index == index) return &index_token_.find(iter->first)->second;

  return nullptr;
}

float Dictionary::cooc_value(const Token& token, int index) const {
  auto index_iter = token_index_.find(token);
  if (index_iter == token_index_.end()) return 0;

  auto cooc_map_iter = cooc_values_.find(index_iter->second);
  if (cooc_map_iter == cooc_values_.end()) return 0;

  int internal_index = -1;
  for (auto iter = cooc_map_iter->second.begin(); iter != cooc_map_iter->second.end(); ++iter)
    if (++internal_index == index) return iter->second;

  return 0;
}

float Dictionary::cooc_value(const Token& token_1, const Token& token_2) const {
  auto index_iter_1 = token_index_.find(token_1);
  if (index_iter_1 == token_index_.end()) return 0;

  auto index_iter_2 = token_index_.find(token_2);
  if (index_iter_2 == token_index_.end()) return 0;

  auto cooc_map_iter_1 = cooc_values_.find(index_iter_1->second);
  if (cooc_map_iter_1 == cooc_values_.end()) return 0;

  auto cooc_map_iter_2 = cooc_map_iter_1->second.find(index_iter_2->second);
  if (cooc_map_iter_2 == cooc_map_iter_1->second.end()) return 0;

  return cooc_map_iter_2->second;
}

const std::unordered_map<int, float>* Dictionary::cooc_info(const Token& token) const {
  auto index_iter = token_index_.find(token);
  if (index_iter == token_index_.end()) return nullptr;

  auto cooc_map_iter = cooc_values_.find(index_iter->second);
  if (cooc_map_iter == cooc_values_.end()) return nullptr;

  return &(cooc_map_iter->second);
}

const DictionaryEntry* Dictionary::entry(const Token& token) const {
  auto find_iter = token_index_.find(token);
  if (find_iter != token_index_.end())
    return &entries_[find_iter->second];
  else
    return nullptr;
}

const DictionaryEntry* Dictionary::entry(int index) const {
  if (index < 0 || index >= entries_.size()) return nullptr;
  return &entries_[index];
}

float Dictionary::CountTopicCoherence(const std::vector<core::Token>& tokens_to_score) {
  float coherence_value = 0.0;
  int k = tokens_to_score.size();
  if (k == 0 || k == 1) return 0.0f;

  // -1 means that find() result == end()
  auto indices = std::vector<int>(k, -1);
  for (int i = 0; i < k; ++i) {
    auto token_index_iter = token_index_.find(tokens_to_score[i]);
    if (token_index_iter == token_index_.end()) continue;
    indices[i] = token_index_iter->second;
  }

  for (int i = 0; i < k - 1; ++i) {
    if (indices[i] == -1) continue;
    auto cooc_map_iter = cooc_values_.find(indices[i]);
    if (cooc_map_iter == cooc_values_.end()) continue;

    for (int j = i; j < k; ++j) {
      if (indices[j] == -1) continue;
      if (tokens_to_score[j].class_id != tokens_to_score[i].class_id) continue;

      auto value_iter = cooc_map_iter->second.find(indices[j]);
      if (value_iter == cooc_map_iter->second.end()) continue;
      coherence_value += static_cast<float>(value_iter->second);
    }
  }

  return 2.0f / (k * (k - 1)) * coherence_value;
}

}  // namespace core
}  // namespace artm
