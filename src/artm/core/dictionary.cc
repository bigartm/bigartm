// Copyright 2014, Additive Regularization of Topic Models.

#include <string>
#include <map>
#include <memory>

#include "artm/core/dictionary.h"

namespace artm {
namespace core {

DictionaryImpl::DictionaryImpl(const artm::DictionaryData& data) {
  if (data.cooc_value_size() == 0) {
    for (int index = 0; index < data.token_size(); ++index) {
      ClassId class_id = data.class_id_size() ? data.class_id(index) : DefaultClass;
      entries_.push_back(DictionaryEntryImpl(Token(class_id, data.token(index)),
        data.token_value(index), data.token_tf(index), data.token_df(index)));

      token_index_.insert(std::make_pair(entries_[index].token(), index));
    }
  } else {
    LOG(ERROR) << "Can't create Dictionary using the cooc part of DictionaryData";
  }
}

void DictionaryImpl::Append(const DictionaryData& data) {
  // DictionaryData should contain only one of cases of data (tokens or cooc information), not both ones
  if (data.cooc_value_size() == 0) {
    // ToDo: MelLain
    // Currectly unsupported option. Open question: how to index cooc tokens if two
    // or more DictionaryData with tokens are given?

    LOG(WARNING) << "Adding new tokens to Dictionary is currently unsupported";
  } else {
    for (int i = 0; i < data.cooc_first_index_size(); ++i) {
      auto first_index_iter = token_index_.find(entries_[data.cooc_first_index(i)].token());
      auto second_index_iter = token_index_.find(entries_[data.cooc_second_index(i)].token());

      // ignore tokens, that are not represented in dictionary entries
      if (first_index_iter != token_index_.end() && second_index_iter != token_index_.end()) {
        auto first_cooc_iter = cooc_values_.find(first_index_iter->second);
        if (first_cooc_iter == cooc_values_.end()) {
          cooc_values_.insert(std::make_pair(first_index_iter->second, std::unordered_map<int, float>()));
          first_cooc_iter = cooc_values_.find(first_index_iter->second);
        }

        // std::map::insert() ignores attempts to write several pairs with same key
        first_cooc_iter->second.insert(std::make_pair(second_index_iter->second,
                                                      data.cooc_value(i)));
      }
    }
  }
}

std::shared_ptr<DictionaryImpl> DictionaryImpl::Duplicate() const {
  return std::shared_ptr<DictionaryImpl>(new DictionaryImpl(*this));
}

void DictionaryImpl::StoreIntoDictionaryData(DictionaryData* data) const {
  for (int i = 0; i < entries_.size(); ++i) {
    data->add_token(entries_[i].token().keyword);
    data->add_class_id(entries_[i].token().class_id);
    data->add_token_value(entries_[i].token_value());
    data->add_token_tf(entries_[i].token_tf());
    data->add_token_df(entries_[i].token_df());
  }
}

const std::unordered_map<int, float>* DictionaryImpl::cooc_info(const Token& token) const {
  auto index_iter = token_index_.find(token);
  if (index_iter == token_index_.end()) return nullptr;

  auto cooc_map_iter = cooc_values_.find(index_iter->second);
  if (cooc_map_iter == cooc_values_.end()) return nullptr;

  return &(cooc_map_iter->second);
}

const DictionaryEntryImpl* DictionaryImpl::entry(const Token& token) const {
  auto find_iter = token_index_.find(token);
  if (find_iter != token_index_.end())
    return &entries_[find_iter->second];
  else
    return nullptr;
}

const DictionaryEntryImpl* DictionaryImpl::entry(int index) const {
  if (index < 0 || index >= entries_.size()) return nullptr;
  return &entries_[index];
}

float DictionaryImpl::CountTopicCoherence(const std::vector<core::Token>& tokens_to_score) {
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

Dictionary::Dictionary(const artm::DictionaryConfig& config) {
  total_items_count_ = config.total_items_count();
  for (int index = 0; index < config.entry_size(); ++index) {
    const ::artm::DictionaryEntry& entry = config.entry(index);
    ClassId class_id;
    if (entry.has_class_id()) class_id = entry.class_id();
    else                      class_id = DefaultClass;
    token_index_.insert(std::make_pair(Token(class_id, entry.key_token()), index));
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
