// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/collection_parser.h"

#include <algorithm>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <utility>

#include "boost/algorithm/string/predicate.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/iostreams/device/mapped_file.hpp"
#include "boost/iostreams/stream.hpp"

#include "glog/logging.h"

#include "artm/core/common.h"
#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"

using boost::iostreams::mapped_file_source;

namespace artm {
namespace core {

CollectionParser::CoocurrenceStatisticsAccumulator::CoocurrenceStatisticsAccumulator(
    const TokenMap& token_info,
    const ::google::protobuf::RepeatedPtrField< ::std::string>& tokens_to_collect)
    : token_info_(token_info),
      tokens_to_collect_(),
      token_coocurrence_(),
      item_tokens_() {
  for (int i = 0; i < tokens_to_collect.size(); ++i) {
    tokens_to_collect_.insert(tokens_to_collect.Get(i));
  }
}

void CollectionParser::CoocurrenceStatisticsAccumulator::AppendTokenId(int token_id) {
  std::string keyword = token_info_.find(token_id)->second.keyword;
  if (tokens_to_collect_.find(keyword) != tokens_to_collect_.end()) {
    item_tokens_.push_back(token_id);
  }
}

void CollectionParser::CoocurrenceStatisticsAccumulator::FlushNewItem() {
  std::sort(item_tokens_.begin(), item_tokens_.end());
  item_tokens_.erase(std::unique(item_tokens_.begin(), item_tokens_.end()),
                                  item_tokens_.end());
  for (size_t first_token_id = 0; first_token_id < item_tokens_.size(); ++first_token_id) {
    for (size_t second_token_id = (first_token_id + 1); second_token_id < item_tokens_.size();
      ++second_token_id) {
      int first_token = item_tokens_[first_token_id];
      int second_token = item_tokens_[second_token_id];
      auto iter = token_coocurrence_.find(std::make_pair(first_token, second_token));
      if (iter == token_coocurrence_.end()) {
        token_coocurrence_.insert(
          std::make_pair(std::make_pair(first_token, second_token), 1));

        // Warn about too large dictionaries.
        if (token_coocurrence_.size() % 1000000 == 0) {
          LOG(WARNING) << "The size of cooccurrence dictionary has reached "
            << token_coocurrence_.size();
        }
      } else {
        iter->second++;
      }
    }
  }

  item_tokens_.clear();
}

void CollectionParser::CoocurrenceStatisticsAccumulator::Export(DictionaryConfig* dictionary) {
  for (auto iter = token_coocurrence_.begin(); iter != token_coocurrence_.end(); ++iter) {
    DictionaryEntry *entry = dictionary->add_entry();
    std::string first_key = token_info_.find(iter->first.first)->second.keyword;
    std::string second_key = token_info_.find(iter->first.second)->second.keyword;
    std::string key = (first_key < second_key) ? (first_key + "~" + second_key)
                                                : (second_key + "~" + first_key);
    entry->set_key_token(key);
    entry->set_items_count(iter->second);
  }
}

CollectionParser::CollectionParser(const ::artm::CollectionParserConfig& config)
    : config_(config) {}



std::shared_ptr<DictionaryConfig> CollectionParser::ParseDocwordBagOfWordsUci(TokenMap* token_map) {
  if (!boost::filesystem::exists(config_.docword_file_path()))
    BOOST_THROW_EXCEPTION(DiskReadException(
      "File " + config_.docword_file_path() + " does not exist."));

  boost::iostreams::stream<mapped_file_source> docword(config_.docword_file_path());

  // Skip all lines starting with "%" and parse N, W, NNZ from the first line after that.
  auto pos = docword.tellg();
  std::string str;
  while (true) {
    pos = docword.tellg();
    std::getline(docword, str);
    if (!boost::starts_with(str.c_str(), "%")) {
      docword.seekg(pos);
      break;
    }

    if (docword.eof()) {
      BOOST_THROW_EXCEPTION(DiskReadException(
        "No content found in" + config_.docword_file_path()));
    }
  }

  int num_docs, num_unique_tokens, num_tokens;
  docword >> num_docs >> num_unique_tokens >> num_tokens;

  if (num_docs <= 0) {
    BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
      "CollectionParser.num_docs (D)", num_docs));
  }

  if (num_unique_tokens <= 0) {
    BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
      "CollectionParser.num_unique_tokens (W)", num_unique_tokens));
  }

  if (num_tokens <= 0) {
    BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
      "CollectionParser.num_tokens (NNZ)", num_tokens));
  }

  if (token_map->empty()) {
    // Autogenerate some tokens
    for (int i = 0; i < num_tokens; ++i) {
      std::string token_keyword = boost::lexical_cast<std::string>(i);
      token_map->insert(std::make_pair(i, CollectionParserTokenInfo(token_keyword)));
    }
  }

  std::unique_ptr<CoocurrenceStatisticsAccumulator> cooc_accum;
  if (config_.has_cooccurrence_file_name()) {
    if (config_.cooccurrence_token_size() == 0) {
      BOOST_THROW_EXCEPTION(InvalidOperation(
        "CollectionParser.cooccurrence_token is empty"));
    } else {
      cooc_accum.reset(new CoocurrenceStatisticsAccumulator(
        *token_map, config_.cooccurrence_token()));
    }
  }

  std::map<int, int> batch_dictionary;
  ::artm::Batch batch;
  ::artm::Item* item = nullptr;
  ::artm::Field* field = nullptr;
  int prev_item_id = -1;

  int64_t total_token_count = 0;
  int64_t total_items_count = 0;
  int token_count_zero = 0;

  int item_id, token_id, token_count;
  for (std::string token; docword >> item_id >> token_id >> token_count;) {
    token_id--;  // convert 1-based to zero-based index

    if (token_map->find(token_id) == token_map->end())  {
      std::stringstream ss;
      ss << "Failed to parse line '" << item_id << " " << (token_id+1) << " " << token_count << "' in "
         << config_.docword_file_path();
      if (token_id == -1) {
        ss << ". wordID column appears to be zero-based in the docword file being parsed. "
           << "UCI format defines wordID column to be unity-based. "
           << "Please, increase wordID by one in your input data.";
      } else {
        ss << ". Token_id value is outside of the expected range.";
      }

      BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException("wordID", token_id, ss.str()));
    }

    if (token_count == 0) {
      token_count_zero++;
      continue;
    }

    if (item_id != prev_item_id) {
      prev_item_id = item_id;
      if (batch.item_size() >= config_.num_items_per_batch()) {
        ::artm::core::BatchHelpers::SaveBatch(batch, config_.target_folder());
        batch.Clear();
        batch_dictionary.clear();
      }

      item = batch.add_item();
      item->set_id(item_id);
      field = item->add_field();

      // Increment statistics
      total_items_count++;
      if (cooc_accum) cooc_accum->FlushNewItem();
      LOG_IF(INFO, total_items_count % 100000 == 0) << total_items_count << " documents parsed.";
    }

    auto iter = batch_dictionary.find(token_id);
    if (iter == batch_dictionary.end()) {
      batch_dictionary.insert(std::make_pair(token_id, batch_dictionary.size()));
      batch.add_token((*token_map)[token_id].keyword);
      iter = batch_dictionary.find(token_id);
    }

    field->add_token_id(iter->second);
    field->add_token_count(token_count);
    if (cooc_accum != nullptr) cooc_accum->AppendTokenId(token_id);

    // Increment statistics
    total_token_count += token_count;
    (*token_map)[token_id].items_count++;
    (*token_map)[token_id].token_count += token_count;
  }

  if (batch.item_size() > 0) {
    ::artm::core::BatchHelpers::SaveBatch(batch, config_.target_folder());
    if (cooc_accum) cooc_accum->FlushNewItem();
  }

  // Craft the dictionary
  auto retval = std::make_shared<DictionaryConfig>();
  retval->set_total_items_count(total_items_count);
  retval->set_total_token_count(total_token_count);

  for (auto& key_value : (*token_map)) {
    artm::DictionaryEntry* entry = retval->add_entry();
    entry->set_key_token(key_value.second.keyword);
    entry->set_class_id(DefaultClass);
    entry->set_token_count(key_value.second.token_count);
    entry->set_items_count(key_value.second.items_count);
    entry->set_value(static_cast<double>(key_value.second.token_count) /
                     static_cast<double>(total_token_count));
  }

  if (config_.has_dictionary_file_name()) {
    ::artm::core::BatchHelpers::SaveMessage(config_.dictionary_file_name(),
                                            config_.target_folder(), *retval);
  }

  // Craft the co-occurence dictionary
  if (cooc_accum != nullptr) {
    DictionaryConfig cooc;
    cooc_accum->Export(&cooc);
    cooc.set_total_items_count(total_items_count);
    ::artm::core::BatchHelpers::SaveMessage(config_.cooccurrence_file_name(),
                                            config_.target_folder(), cooc);
  }

  LOG_IF(WARNING, token_count_zero > 0) << "Found " << token_count_zero << " tokens with zero "
                                        << "occurrencies. All these tokens were ignored.";

  return retval;
}

CollectionParser::TokenMap CollectionParser::ParseVocabBagOfWordsUci() {
  if (!boost::filesystem::exists(config_.vocab_file_path()))
    BOOST_THROW_EXCEPTION(DiskReadException(
    "File " + config_.vocab_file_path() + " does not exist."));

  boost::iostreams::stream<mapped_file_source> vocab(config_.vocab_file_path());

  TokenMap token_info;
  int token_id = 0;
  for (std::string token; vocab >> token;) {
    token_info.insert(std::make_pair(token_id, CollectionParserTokenInfo(token)));
    token_id++;
  }

  return token_info;
}

CollectionParser::TokenMap CollectionParser::ParseVocabMatrixMarket() {
  bool has_vocab = config_.has_vocab_file_path();
  if (has_vocab && !boost::filesystem::exists(config_.vocab_file_path())) {
    LOG(WARNING) << "File " + config_.vocab_file_path() + " does not exist.";
  }

  TokenMap token_info;
  if (has_vocab) {
    boost::iostreams::stream<mapped_file_source> vocab(config_.vocab_file_path());

    int token_id, token_count;
    for (std::string token; vocab >> token_id >> token >> token_count;) {
      // token_count is ignored --- it will be re-calculated based on the docword file.
      token_info.insert(std::make_pair(token_id, CollectionParserTokenInfo(token)));
    }
  }

  return token_info;  // empty if no input file had been provided
}

std::shared_ptr<DictionaryConfig> CollectionParser::Parse() {
  TokenMap token_map;
  switch (config_.format()) {
    case CollectionParserConfig_Format_BagOfWordsUci:
      token_map = ParseVocabBagOfWordsUci();
      return ParseDocwordBagOfWordsUci(&token_map);

    case CollectionParserConfig_Format_MatrixMarket:
      token_map = ParseVocabMatrixMarket();
      return ParseDocwordBagOfWordsUci(&token_map);

    default:
      BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
        "CollectionParserConfig.format", config_.format()));
  }
}

}  // namespace core
}  // namespace artm
