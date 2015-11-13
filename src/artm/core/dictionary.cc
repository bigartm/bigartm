// Copyright 2014, Additive Regularization of Topic Models.

#include <string>
#include <map>
#include <memory>

#include "artm/core/dictionary.h"

namespace artm {
namespace core {

Dictionary::Dictionary(const artm::DictionaryData& data) {
  if (data.cooc_value_size() == 0) {
    for (int index = 0; index < data.token_size(); ++index) {
      ClassId class_id = data.class_id_size() ? data.class_id(index) : DefaultClass;
      entries_.push_back(DictionaryEntryImpl(Token(class_id, data.token(index)),
        data.token_value(index), data.token_tf(index), data.token_df(index)));
    
      token_index_.insert(std::make_pair(entries_[index].token(), index));
      index_token_.insert(std::make_pair(index, entries_[index].token()));
    }
  } else {
    LOG(ERROR) << "Can't create Dictionary using the cooc part of DictionaryData";
  }
}

void Dictionary::Append(const DictionaryData& data) {
  if (data.cooc_value_size() == 0) {
    // ToDo: MelLain
    // Currectly unsupported option. Open question: how to index cooc tokens if two
    // or more DictionaryData eith tokens are given?

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

std::shared_ptr<Dictionary> Dictionary::Duplicate() const {
  return std::shared_ptr<Dictionary>(new Dictionary(*this));
}

void Dictionary::StoreIntoDictionaryData(DictionaryData* data) const {
  for (int i = 0; i < entries_.size(); ++i) {
    data->add_token(entries_[i].token().keyword);
    data->add_class_id(entries_[i].token().class_id);
    data->add_token_value(entries_[i].token_value());
    data->add_token_tf(entries_[i].token_tf());
    data->add_token_df(entries_[i].token_df());
  }
}

const std::unordered_map<int, float>* Dictionary::cooc_info(const Token& token) const {
  auto index_iter = token_index_.find(token);
  if (index_iter == token_index_.end()) return nullptr;

  auto cooc_map_iter = cooc_values_.find(index_iter->second);
  if (cooc_map_iter == cooc_values_.end()) return nullptr;

  return &(cooc_map_iter->second);
}

const DictionaryEntryImpl* Dictionary::entry(const Token& token) const {
  auto find_iter = token_index_.find(token);
  if (find_iter != token_index_.end())
    return &entries_[find_iter->second];
  else
    return nullptr;
}

const DictionaryEntryImpl* Dictionary::entry(int index) const {
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
