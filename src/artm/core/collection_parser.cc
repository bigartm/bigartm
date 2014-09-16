// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/collection_parser.h"

#include <algorithm>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <set>
#include <utility>

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

namespace {
  struct CollectionParserTokenInfo {
    explicit CollectionParserTokenInfo()
        : keyword(), token_count(0), items_count(0) {}
    explicit CollectionParserTokenInfo(std::string keyword_)
        : keyword(keyword_), token_count(0), items_count(0) {}

    std::string keyword;
    int token_count;
    int items_count;
  };

  class CoocurrenceStatisticsAccumulator {
   public:
    CoocurrenceStatisticsAccumulator(
        const std::vector<std::unique_ptr<CollectionParserTokenInfo>>& token_info,
        const ::google::protobuf::RepeatedPtrField< ::std::string>& tokens_to_collect)
        : token_info_(token_info),
          tokens_to_collect_(),
          token_coocurrence_(),
          item_tokens_() {
      for (int i = 0; i < tokens_to_collect.size(); ++i) {
        tokens_to_collect_.insert(tokens_to_collect.Get(i));
      }
    }

    void AppendTokenId(int token_id) {
      if (tokens_to_collect_.find(token_info_[token_id]->keyword) != tokens_to_collect_.end()) {
        item_tokens_.push_back(token_id);
      }
    }

    void FlushNewItem() {
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

    void Export(DictionaryConfig* dictionary) {
      for (auto iter = token_coocurrence_.begin(); iter != token_coocurrence_.end(); ++iter) {
        DictionaryEntry *entry = dictionary->add_entry();
        std::string first_key = token_info_[iter->first.first]->keyword;
        std::string second_key = token_info_[iter->first.second]->keyword;
        std::string key = (first_key < second_key) ? (first_key + "~" + second_key)
                                                   : (second_key + "~" + first_key);
        entry->set_key_token(key);
        entry->set_items_count(iter->second);
      }
    }

   private:
    const std::vector<std::unique_ptr<CollectionParserTokenInfo>>& token_info_;
    std::set<std::string> tokens_to_collect_;
    std::map<std::pair<int, int>, int> token_coocurrence_;
    std::vector<int> item_tokens_;
  };
}  // namespace

CollectionParser::CollectionParser(const ::artm::CollectionParserConfig& config)
    : config_(config) {}

std::shared_ptr<DictionaryConfig> CollectionParser::ParseBagOfWordsUci() {
  if (!boost::filesystem::exists(config_.vocab_file_path()))
    BOOST_THROW_EXCEPTION(DiskReadException(
      "File " + config_.vocab_file_path() + " does not exist."));

  if (!boost::filesystem::exists(config_.docword_file_path()))
    BOOST_THROW_EXCEPTION(DiskReadException(
      "File " + config_.docword_file_path() + " does not exist."));

  boost::iostreams::stream<mapped_file_source> vocab(config_.vocab_file_path());
  boost::iostreams::stream<mapped_file_source> docword(config_.docword_file_path());

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

  std::vector<std::unique_ptr<CollectionParserTokenInfo>> token_info;
  token_info.reserve(num_unique_tokens);
  std::unique_ptr<CoocurrenceStatisticsAccumulator> cooc_accum;
  if (config_.has_cooccurrence_file_name()) {
    if (config_.cooccurrence_token_size() == 0) {
      BOOST_THROW_EXCEPTION(InvalidOperation(
        "CollectionParser.cooccurrence_token is empty"));
    } else {
      cooc_accum.reset(new CoocurrenceStatisticsAccumulator(
        token_info, config_.cooccurrence_token()));
    }
  }

  for (std::string token; vocab >> token;) {
    if (static_cast<int>(token_info.size()) >= num_unique_tokens) {
      std::stringstream ss;
      ss << config_.vocab_file_path() << " contains too many tokens. "
         << "Expected number is " << num_unique_tokens
         << ", as it is specified in " << config_.docword_file_path()
         << "). Last read word was '" << token_info.back()->keyword << "'.";

      BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
        "CollectionParser.num_unique_tokens (W)", (num_unique_tokens + 1), ss.str()));
    }

    token_info.push_back(std::unique_ptr<CollectionParserTokenInfo>(
      new CollectionParserTokenInfo(token)));
  }

  if (static_cast<int>(token_info.size()) != num_unique_tokens) {
    std::stringstream ss;
    ss << config_.vocab_file_path() << " contains not enough tokens. "
         << "Expected number is " << num_unique_tokens
         << ", as it is specified in " << config_.docword_file_path()
         << "). Last read word was '" << token_info.back()->keyword << "'.";

    BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
      "CollectionParser.num_unique_tokens (W)", (token_info.size() + 1), ss.str()));
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
    if ((token_id <= 0) || (token_id > static_cast<int>(token_info.size()))) {
      std::stringstream ss;
      ss << "Failed to parse line '" << item_id << " " << token_id << " " << token_count << "' in "
         << config_.docword_file_path();
      if (token_id == 0) {
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

    token_id--;  // convert 1-based to zero-based index

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
      batch.add_token(token_info[token_id]->keyword);
      iter = batch_dictionary.find(token_id);
    }

    field->add_token_id(iter->second);
    field->add_token_count(token_count);
    if (cooc_accum != nullptr) cooc_accum->AppendTokenId(token_id);

    // Increment statistics
    total_token_count += token_count;
    token_info[token_id]->items_count++;
    token_info[token_id]->token_count += token_count;
  }

  if (batch.item_size() > 0) {
    ::artm::core::BatchHelpers::SaveBatch(batch, config_.target_folder());
    if (cooc_accum) cooc_accum->FlushNewItem();
  }

  // Craft the dictionary
  auto retval = std::make_shared<DictionaryConfig>();
  retval->set_total_items_count(total_items_count);
  retval->set_total_token_count(total_token_count);
  for (int token_id = 0; token_id < static_cast<int>(token_info.size()); ++token_id) {
    artm::DictionaryEntry* entry = retval->add_entry();
    entry->set_key_token(token_info[token_id]->keyword);
    entry->set_class_id(DefaultClass);
    entry->set_token_count(token_info[token_id]->token_count);
    entry->set_items_count(token_info[token_id]->items_count);
    entry->set_value(static_cast<double>(token_info[token_id]->token_count) /
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

std::shared_ptr<DictionaryConfig> CollectionParser::Parse() {
  switch (config_.format()) {
    case CollectionParserConfig_Format_BagOfWordsUci:
      return ParseBagOfWordsUci();

    default:
      BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
        "CollectionParserConfig.format", config_.format()));
  }
}

}  // namespace core
}  // namespace artm
