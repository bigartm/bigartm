// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/collection_parser.h"

#include <algorithm>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <utility>

#include "boost/algorithm/string.hpp"
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
    const ::google::protobuf::RepeatedPtrField< ::std::string>& tokens_to_collect,
    const ::google::protobuf::RepeatedPtrField< ::std::string>& class_ids_to_collect)
    : token_info_(token_info),
      tokens_to_collect_(),
      token_coocurrence_(),
      item_tokens_() {
  if (tokens_to_collect.size() == 0) {
    for (auto token : token_info) {
      tokens_to_collect_.insert(Token(token.second.class_id, token.second.keyword));
      if (tokens_to_collect_.size() % 5000 == 0) {
        LOG(INFO) << "The number of unique single tokens cooccurrence dictionary has reached "
            << tokens_to_collect_.size();
      }
    }
  } else {
    for (int i = 0; i < tokens_to_collect.size(); ++i) {
      tokens_to_collect_.insert(Token(class_ids_to_collect.Get(i), tokens_to_collect.Get(i)));
    }
  }
}

void CollectionParser::CoocurrenceStatisticsAccumulator::AppendTokenId(int token_id) {
  Token token = Token(token_info_.find(token_id)->second.class_id,
                      token_info_.find(token_id)->second.keyword);
  if (tokens_to_collect_.find(token) != tokens_to_collect_.end()) {
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

void CollectionParser::CoocurrenceStatisticsAccumulator::Export(std::shared_ptr<DictionaryConfig> dictionary) {
  dictionary->clear_cooc_entries();
  artm::DictionaryCoocurenceEntries* cooc_entries = dictionary->mutable_cooc_entries();
  for (auto iter = token_coocurrence_.begin(); iter != token_coocurrence_.end(); ++iter) {
    cooc_entries->add_first_index(iter->first.first);
    cooc_entries->add_second_index(iter->first.second);
    cooc_entries->add_value(iter->second);
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
    for (int i = 0; i < num_unique_tokens; ++i) {
      std::string token_keyword = boost::lexical_cast<std::string>(i);
      token_map->insert(std::make_pair(i, CollectionParserTokenInfo(token_keyword, DefaultClass)));
    }
  }

  std::unique_ptr<CoocurrenceStatisticsAccumulator> cooc_accum;
  if (config_.gather_cooc()) {
    cooc_accum.reset(new CoocurrenceStatisticsAccumulator(
      *token_map, config_.cooccurrence_token(), config_.cooccurrence_class_id()));
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
  int index = 1;
  std::getline(docword, str);  // skip end of previos line

  while (!docword.eof()) {
    std::getline(docword, str);
    boost::algorithm::trim(str);
    ++index;
    if (str.empty()) continue;

    std::vector<std::string> strs;
    boost::split(strs, str, boost::is_any_of("\t "));
    if (strs.size() != 3) {
      std::stringstream ss;
      ss << "Error at line" << index << " or " << index + 2 << ", file " << config_.docword_file_path()
         << ". Expected format: item_id token_id n_wd";
      BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    }

    item_id = std::stoi(strs[0]);
    token_id = std::stoi(strs[1]);
    token_count = std::stoi(strs[2]);

    if (config_.use_unity_based_indices())
      token_id--;  // convert 1-based to zero-based index

    if (token_map->find(token_id) == token_map->end())  {
      std::stringstream ss;
      ss << "Failed to parse line '" << item_id << " " << (token_id + 1) << " " << token_count << "' in "
         << config_.docword_file_path();
      if (token_id == -1) {
        ss << ". wordID column appears to be zero-based in the docword file being parsed. "
           << "UCI format defines wordID column to be unity-based. "
           << "Please, set CollectionParserConfig.use_unity_based_indices=false "
           << "or increase wordID by one in your input data";
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
      batch.add_class_id((*token_map)[token_id].class_id);
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
  retval->set_name(config_.dictionary_file_name());
  retval->set_total_items_count(total_items_count);
  retval->set_total_token_count(total_token_count);

  for (auto& key_value : (*token_map)) {
    artm::DictionaryEntry* entry = retval->add_entry();
    entry->set_key_token(key_value.second.keyword);
    entry->set_class_id(key_value.second.class_id);
    entry->set_token_count(key_value.second.token_count);
    entry->set_items_count(key_value.second.items_count);
    entry->set_value(static_cast<double>(key_value.second.token_count) /
                     static_cast<double>(total_token_count));
  }

  // Craft the co-occurence part of dictionary
  if (cooc_accum != nullptr) {
    cooc_accum->Export(retval);
  }

  if (config_.has_dictionary_file_name()) {
    ::artm::core::BatchHelpers::SaveMessage(config_.dictionary_file_name(),
                                            config_.target_folder(), *retval);
  }

  LOG_IF(WARNING, token_count_zero > 0) << "Found " << token_count_zero << " tokens with zero "
                                        << "occurrencies. All these tokens were ignored.";

  return retval;
}

std::shared_ptr<DictionaryConfig> CollectionParser::ParseCooccurrenceData(TokenMap* token_map) {
  if (!boost::filesystem::exists(config_.docword_file_path()))
    BOOST_THROW_EXCEPTION(DiskReadException(
      "File " + config_.docword_file_path() + " does not exist."));

  boost::iostreams::stream<mapped_file_source> user_cooc_data(config_.docword_file_path());

  // Craft the dictionary
  auto retval = std::make_shared<DictionaryConfig>();
  retval->set_name(config_.dictionary_file_name());
  retval->set_total_items_count(0);
  retval->set_total_token_count(0);

  for (auto& key_value : (*token_map)) {
    artm::DictionaryEntry* entry = retval->add_entry();
    entry->set_key_token(key_value.second.keyword);
    entry->set_class_id(key_value.second.class_id);
    entry->set_token_count(0);
    entry->set_items_count(0);
    entry->set_value(0);
  }

  // Craft the co-occurence part of dictionary
  int index = 0;
  std::string str;
  bool last_line = false;
  retval->clear_cooc_entries();
  artm::DictionaryCoocurenceEntries* cooc_entries = retval->mutable_cooc_entries();
  while (!user_cooc_data.eof()) {
    if (last_line) {
      std::stringstream ss;
      ss << "Empty pair of tokens at line " << index << ", file " << config_.docword_file_path();
      BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    }
    std::getline(user_cooc_data, str);
    ++index;
    boost::algorithm::trim(str);
    if (str.empty()) {
      last_line = true;
      continue;
    }

    std::vector<std::string> strs;
    boost::split(strs, str, boost::is_any_of("\t "));
    if (strs.size() < 3) {
      std::stringstream ss;
      ss << "Error at line " << index << ", file " << config_.docword_file_path()
         << ". Expected format: <token_id_1> <token_id_2> {<cooc_value>}";
      BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    }

    if (strs.size() != 3) {
      std::stringstream ss;
      ss << "Error at line " << index << ", file " << config_.docword_file_path()
         << ". Number of values in all lines should be equal to 3";
      BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    }

    cooc_entries->add_first_index(std::stoi(strs[0]));
    cooc_entries->add_second_index(std::stoi(strs[1]));
    cooc_entries->add_value(std::stof(strs[2]));
  }

  if (config_.has_dictionary_file_name()) {
    ::artm::core::BatchHelpers::SaveMessage(config_.dictionary_file_name(),
                                            config_.target_folder(), *retval);
  }

  return retval;
}

CollectionParser::TokenMap CollectionParser::ParseVocabBagOfWordsUci() {
  if (!boost::filesystem::exists(config_.vocab_file_path()))
    BOOST_THROW_EXCEPTION(DiskReadException(
    "File " + config_.vocab_file_path() + " does not exist."));

  boost::iostreams::stream<mapped_file_source> vocab(config_.vocab_file_path());

  std::map<Token, int> token_to_token_id;

  TokenMap token_info;
  std::string str;
  int token_id = 0;
  while (!vocab.eof()) {
    std::getline(vocab, str);
    if (vocab.eof())
      break;

    boost::algorithm::trim(str);
    if (str.empty()) {
      std::stringstream ss;
      ss << "Empty token at line " << (token_id + 1) << ", file " << config_.vocab_file_path();
      BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    }

    std::vector<std::string> strs;
    boost::split(strs, str, boost::is_any_of("\t "));
    if ((strs.size() == 0) || (strs.size() > 2)) {
      std::stringstream ss;
      ss << "Error at line " << (token_id + 1) << ", file " << config_.vocab_file_path()
         << ". Expected format: <token> [<class_id>]";
      BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    }

    ClassId class_id = (strs.size() == 2) ? strs[1] : DefaultClass;
    Token token(class_id, strs[0]);

    if (token_to_token_id.find(token) != token_to_token_id.end()) {
      std::stringstream ss;
      ss << "Token (" << token.keyword << ", " << token.class_id << "' found twice, lines "
         << (token_to_token_id.find(token)->second + 1)
         << " and " << (token_id + 1) << ", file " << config_.vocab_file_path();
      BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    }

    token_info.insert(std::make_pair(token_id, CollectionParserTokenInfo(token.keyword, token.class_id)));
    token_to_token_id.insert(std::make_pair(token, token_id));
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
      token_info.insert(std::make_pair(token_id, CollectionParserTokenInfo(token, DefaultClass)));
    }
  }

  return token_info;  // empty if no input file had been provided
}

// ToDo: Collect token cooccurrence in BatchCollector, and export it in ParseVowpalWabbit().
class CollectionParser::BatchCollector {
 private:
  Item *item_;
  Batch batch_;
  std::map<Token, int> local_map_;
  std::map<Token, CollectionParserTokenInfo> global_map_;
  int64_t total_token_count_;
  int64_t total_items_count_;

  void StartNewItem() {
    item_ = batch_.add_item();
    item_->add_field();
    total_items_count_++;
  }

 public:
  BatchCollector() : item_(nullptr), total_token_count_(0), total_items_count_(0) {}

  void Record(Token token, int token_count) {
    if (global_map_.find(token) == global_map_.end())
      global_map_.insert(std::make_pair(token, CollectionParserTokenInfo(token.keyword, token.class_id)));
    if (local_map_.find(token) == local_map_.end()) {
      local_map_.insert(std::make_pair(token, batch_.token_size()));
      batch_.add_token(token.keyword);
      batch_.add_class_id(token.class_id);
    }

    CollectionParserTokenInfo& token_info = global_map_[token];
    int local_token_id = local_map_[token];

    if (item_ == nullptr) StartNewItem();

    Field* field = item_->mutable_field(0);
    field->add_token_id(local_token_id);
    field->add_token_count(token_count);

    token_info.items_count++;
    token_info.token_count += token_count;
    total_token_count_ += token_count;
  }

  void FinishItem(int item_id, std::string item_title) {
    if (item_ == nullptr) StartNewItem();  // this item fill be empty;

    item_->set_id(item_id);
    item_->set_title(item_title);

    LOG_IF(INFO, total_items_count_ % 100000 == 0) << total_items_count_ << " documents parsed.";


    // Item is already included in the batch;
    // Set item_ to nullptr to finish it; then next Record() will create a new item;
    item_ = nullptr;
  }

  Batch FinishBatch() {
    Batch batch;
    batch.Swap(&batch_);
    local_map_.clear();
    return batch;
  }

  const Batch& batch() { return batch_; }

  std::shared_ptr<DictionaryConfig> ExportDictionaryConfig() {
    // Craft the dictionary
    auto retval = std::make_shared<DictionaryConfig>();
    retval->set_total_items_count(total_items_count_);
    retval->set_total_token_count(total_token_count_);

    for (auto& key_value : global_map_) {
      artm::DictionaryEntry* entry = retval->add_entry();
      entry->set_key_token(key_value.second.keyword);
      entry->set_class_id(key_value.second.class_id);
      entry->set_token_count(key_value.second.token_count);
      entry->set_items_count(key_value.second.items_count);
      entry->set_value(static_cast<double>(key_value.second.token_count) /
        static_cast<double>(total_token_count_));
    }

    return retval;
  }
};

std::shared_ptr<DictionaryConfig> CollectionParser::ParseVowpalWabbit() {
  BatchCollector batch_collector;

  if (!boost::filesystem::exists(config_.docword_file_path()))
    BOOST_THROW_EXCEPTION(DiskReadException(
    "File " + config_.docword_file_path() + " does not exist."));

  boost::iostreams::stream<mapped_file_source> docword(config_.docword_file_path());
  std::string str;
  int line_no = 0;
  while (!docword.eof()) {
    std::getline(docword, str);
    line_no++;
    if (docword.eof())
      break;

    std::vector<std::string> strs;
    boost::split(strs, str, boost::is_any_of(" \t\r"));

    if (strs.size() <= 1) {
      std::stringstream ss;
      ss << "Error in " << config_.docword_file_path() << ":" << line_no << ", too few entries: " << str;
      BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    }

    std::string item_title = strs[0];

    ClassId class_id = DefaultClass;
    for (int elem_index = 1; elem_index < strs.size(); ++elem_index) {
      std::string elem = strs[elem_index];
      if (elem.size() == 0)
        continue;
      if (elem[0] == '|') {
        class_id = elem.substr(1);
        continue;
      }

      int token_count = 1;
      std::string token = elem;
      size_t split_index = elem.find(':');
      if (split_index != std::string::npos) {
        if (split_index == 0 || split_index == (elem.size() - 1)) {
          std::stringstream ss;
          ss << "Error in " << config_.docword_file_path() << ":" << line_no
             << ", entries can not start or end with colon: " << elem;
          BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
        }
        token = elem.substr(0, split_index);
        std::string token_occurences_string = elem.substr(split_index + 1);
        try {
          token_count = boost::lexical_cast<int>(token_occurences_string);
        }
        catch (boost::bad_lexical_cast &) {
          std::stringstream ss;
          ss << "Error in " << config_.docword_file_path() << ":" << line_no
             << ", can not parse integer number of occurences: " << elem;
          BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
        }
      }

      batch_collector.Record(artm::core::Token(class_id, token), token_count);
    }

    batch_collector.FinishItem(line_no, item_title);
    if (batch_collector.batch().item_size() >= config_.num_items_per_batch()) {
      ::artm::core::BatchHelpers::SaveBatch(batch_collector.FinishBatch(), config_.target_folder());
    }
  }

  if (batch_collector.batch().item_size() > 0) {
    ::artm::core::BatchHelpers::SaveBatch(batch_collector.FinishBatch(), config_.target_folder());
  }

  std::shared_ptr<DictionaryConfig> retval = batch_collector.ExportDictionaryConfig();

  if (config_.has_dictionary_file_name()) {
    ::artm::core::BatchHelpers::SaveMessage(config_.dictionary_file_name(),
      config_.target_folder(), *retval);
  }

  retval->set_name(config_.dictionary_file_name());
  return retval;
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

    case CollectionParserConfig_Format_VowpalWabbit:
      return ParseVowpalWabbit();

    case CollectionParserConfig_Format_Cooccurrence:
      token_map = ParseVocabBagOfWordsUci();
      return ParseCooccurrenceData(&token_map);

    default:
      BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
        "CollectionParserConfig.format", config_.format()));
  }
}

}  // namespace core
}  // namespace artm
