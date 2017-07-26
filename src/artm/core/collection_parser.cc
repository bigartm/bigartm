// Copyright 2017, Additive Regularization of Topic Models.

#include "artm/core/collection_parser.h"

#include <algorithm>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <utility>
#include <unordered_map>
#include <iostream>  // NOLINT
#include <future>  // NOLINT

#include "boost/algorithm/string.hpp"
#include "boost/algorithm/string/predicate.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

#include "glog/logging.h"

#include "artm/utility/ifstream_or_cin.h"
#include "artm/utility/progress_printer.h"

#include "artm/core/common.h"
#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/protobuf_helpers.h"

using ::artm::utility::ifstream_or_cin;

namespace artm {
namespace core {

BatchNameGenerator::BatchNameGenerator(int length, bool use_guid_name)
    : length_(length)
    , next_name_(std::string())
    , use_guid_name_(use_guid_name) {
  next_name_ = std::string(length, 'a');
}

std::string BatchNameGenerator::next_name(const Batch& batch) {
  if (use_guid_name_) {
    return batch.id();
  }

  std::string old_next_name = next_name_;

  for (int i = length_ - 1; i >= 0; --i) {
    if (next_name_[i] != 'z') {
      next_name_[i] += 1;
      break;
    } else {
      if (i == 0) {
          BOOST_THROW_EXCEPTION(InvalidOperation("Parser can't create more batches"));
      } else {
          next_name_[i] = 'a';
      }
    }
  }

  return old_next_name;
}

static bool useClassId(const ClassId& class_id, const CollectionParserConfig& config) {
  if (config.class_id_size() == 0) {
    return true;
  }
  if (class_id.empty() || class_id == DefaultClass) {
    return is_member(std::string(), config.class_id()) || is_member(DefaultClass, config.class_id());
  }
  return is_member(class_id, config.class_id());
}

CollectionParser::CollectionParser(const ::artm::CollectionParserConfig& config) : config_(config) { }

CollectionParserInfo CollectionParser::ParseDocwordBagOfWordsUci(TokenMap* token_map) {
  BatchNameGenerator batch_name_generator(kBatchNameLength,
    config_.name_type() == CollectionParserConfig_BatchNameType_Guid);
  ifstream_or_cin stream_or_cin(config_.docword_file_path());
  std::istream& docword = stream_or_cin.get_stream();
  utility::ProgressPrinter progress(stream_or_cin.size());

  // Skip all lines starting with "%" and parse N, W, NNZ from the first line after that.
  auto pos = docword.tellg();
  std::string str;
  while (true) {
    pos = docword.tellg();
    std::getline(docword, str);
    if (!boost::starts_with(str.c_str(), "%")) {
      // FIXME (JeanPaulShapo) there can be failures when reading from standard input
      // there's no guarantee that seekg successfully move stream pointer, especially with std::cin
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

  std::map<int, int> batch_dictionary;
  ::artm::Batch batch;
  ::artm::Item* item = nullptr;
  int prev_item_id = -1;

  float total_token_weight = 0;
  int64_t total_items_count = 0;
  int64_t token_weight_zero = 0;
  int64_t total_triples_count = 0;
  int64_t num_batches = 0;

  int item_id, token_id;
  float token_weight;
  int line_no = 1;
  std::getline(docword, str);  // skip end of previous line

  while (!docword.eof()) {
    std::getline(docword, str);
    boost::algorithm::trim(str);
    ++line_no;
    progress.Set(docword.tellg());
    if (str.empty()) {
      continue;
    }

    std::vector<std::string> strs;
    boost::split(strs, str, boost::is_any_of("\t "));
    if (strs.size() != 3) {
      std::stringstream ss;
      ss << "Error at line" << line_no << " or " << line_no + 2 << ", file " << config_.docword_file_path()
         << ". Expected format: item_id token_id n_wd";
      BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    }

    item_id = std::stoi(strs[0]);
    token_id = std::stoi(strs[1]);
    token_weight = std::stof(strs[2]);

    if (config_.use_unity_based_indices()) {
      token_id--;  // convert 1-based to zero-based index
    }

    if (token_map->find(token_id) == token_map->end())  {
      std::stringstream ss;
      ss << "Failed to parse line '" << item_id << " " << (token_id + 1) << " " << token_weight << "' in "
         << config_.docword_file_path();
      if (token_id == -1 && config_.use_unity_based_indices()) {
        ss << ". wordID column appears to be zero-based in the docword file being parsed. "
           << "UCI format defines wordID column to be unity-based. "
           << "Please, set CollectionParserConfig.use_unity_based_indices=false "
           << "or increase wordID by one in your input data";
      } else {
        ss << ". Token_id value is outside of the expected range.";
      }

      BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException("wordID", token_id, ss.str()));
    }

    if (token_weight == 0.0f) {
      token_weight_zero++;
      continue;
    }

    if (item_id != prev_item_id) {
      prev_item_id = item_id;
      if (batch.item_size() >= config_.num_items_per_batch()) {
        batch.set_id(boost::lexical_cast<std::string>(boost::uuids::random_generator()()));
        ::artm::core::Helpers::SaveBatch(batch, config_.target_folder(), batch_name_generator.next_name(batch));
        num_batches++;
        batch.Clear();
        batch_dictionary.clear();
      }

      item = batch.add_item();
      item->set_id(item_id);

      // Increment statistics
      total_items_count++;
      LOG_IF(INFO, total_items_count % 100000 == 0) << total_items_count << " documents parsed.";
    }

    // Skip token when it is not among modalities that user has requested to parse
    if (!useClassId((*token_map)[token_id].class_id, config_)) {
      continue;
    }

    auto iter = batch_dictionary.find(token_id);
    if (iter == batch_dictionary.end()) {
      batch_dictionary.insert(std::make_pair(token_id, static_cast<int>(batch_dictionary.size())));
      batch.add_token((*token_map)[token_id].keyword);
      batch.add_class_id((*token_map)[token_id].class_id);
      iter = batch_dictionary.find(token_id);
    }

    item->add_token_id(iter->second);
    item->add_token_weight(token_weight);

    // Increment statistics
    total_token_weight += token_weight;
    total_triples_count++;
    (*token_map)[token_id].items_count++;
    (*token_map)[token_id].token_weight += token_weight;
  }

  if (batch.item_size() > 0) {
    batch.set_id(boost::lexical_cast<std::string>(boost::uuids::random_generator()()));
    ::artm::core::Helpers::SaveBatch(batch, config_.target_folder(), batch_name_generator.next_name(batch));
    num_batches++;
  }

  LOG_IF(WARNING, token_weight_zero > 0) << "Found " << token_weight_zero << " tokens with zero "
    << "occurrencies. All these tokens were ignored.";
  { // Count number of missed tokens
    auto missed_tokens = std::count_if(token_map->begin(), token_map->end(),
        [](const TokenMap::value_type &token_info) {
          return !(token_info.second.items_count);
        });
    // Warn if some tokens are not included in any document
    LOG_IF(WARNING, missed_tokens) << missed_tokens << " aren't present in parsed collection";
  }
  // Warn if possible number of parsed documents doesn't equal to expected one
  LOG_IF(WARNING, num_docs != total_items_count) << "Expected " << num_docs << " documents to parse, found "
    << total_items_count;
  // Warn if number of triples doesn't equal to expected one
  LOG_IF(WARNING, num_tokens != total_triples_count) << "Expected " << num_tokens
    << " triples describing collection, found " << total_triples_count;

  CollectionParserInfo parser_info;
  parser_info.set_num_items(total_items_count);
  parser_info.set_num_batches(num_batches);
  parser_info.set_dictionary_size(token_map->size());
  parser_info.set_num_tokens(total_triples_count);
  parser_info.set_total_token_weight(total_token_weight);
  return parser_info;
}

CollectionParser::TokenMap CollectionParser::ParseVocabBagOfWordsUci() {
  ifstream_or_cin stream_or_cin(config_.vocab_file_path());
  std::istream& vocab = stream_or_cin.get_stream();

  std::map<Token, int> token_to_token_id;

  TokenMap token_info;
  std::string str;
  int token_id = 0;
  while (!vocab.eof()) {
    std::getline(vocab, str);
    if (vocab.eof() && str.empty()) {
      break;
    }

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

  TokenMap token_info;
  if (has_vocab) {
    ifstream_or_cin stream_or_cin(config_.vocab_file_path());
    std::istream& vocab = stream_or_cin.get_stream();

    int token_id;
    float token_weight;
    for (std::string token; vocab >> token_id >> token >> token_weight;) {
      // token_weight is ignored --- it will be re-calculated based on the docword file.
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
  float total_token_weight_;
  int64_t total_items_count_;
  int64_t total_tokens_count_;

  void StartNewItem() {
    item_ = batch_.add_item();
    total_items_count_++;
  }

 public:
  BatchCollector() : item_(nullptr), total_token_weight_(0), total_items_count_(0), total_tokens_count_(0) {
    batch_.set_id(boost::lexical_cast<std::string>(boost::uuids::random_generator()()));
  }

  void Record(Token token, float token_weight) {
    if (global_map_.find(token) == global_map_.end()) {
      global_map_.insert(std::make_pair(token, CollectionParserTokenInfo(token.keyword, token.class_id)));
    }
    if (local_map_.find(token) == local_map_.end()) {
      local_map_.insert(std::make_pair(token, batch_.token_size()));
      batch_.add_token(token.keyword);
      batch_.add_class_id(token.class_id);
    }

    CollectionParserTokenInfo& token_info = global_map_[token];
    int local_token_id = local_map_[token];

    if (item_ == nullptr) {
      StartNewItem();
    }

    item_->add_token_id(local_token_id);
    item_->add_token_weight(token_weight);

    token_info.items_count++;
    token_info.token_weight += token_weight;
    total_token_weight_ += token_weight;
    total_tokens_count_ += 1;
  }

  void FinishItem(int item_id, std::string item_title) {
    if (item_ == nullptr) {
      StartNewItem();  // this item fill be empty;
    }

    item_->set_id(item_id);
    item_->set_title(item_title);

    LOG_IF(INFO, total_items_count_ % 100000 == 0) << total_items_count_ << " documents parsed.";

    // Item is already included in the batch;
    // Set item_ to nullptr to finish it; then next Record() will create a new item;
    item_ = nullptr;
  }

  Batch FinishBatch(CollectionParserInfo* info) {
    info->set_num_items(info->num_items() + total_items_count_);
    info->set_num_tokens(info->num_tokens() + total_tokens_count_);
    info->set_total_token_weight(info->total_token_weight() + total_token_weight_);
    info->set_num_batches(info->num_batches() + 1);

    Batch batch;
    batch.Swap(&batch_);
    local_map_.clear();
    return batch;
  }

  const Batch& batch() { return batch_; }
};

CollectionParserInfo CollectionParser::ParseVowpalWabbit() {
  BatchNameGenerator batch_name_generator(kBatchNameLength,
    config_.name_type() == CollectionParserConfig_BatchNameType_Guid);
  ifstream_or_cin stream_or_cin(config_.docword_file_path());
  std::istream& docword = stream_or_cin.get_stream();
  utility::ProgressPrinter progress(stream_or_cin.size());

  auto config = config_;

  std::mutex lock;
  int global_line_no = 0;

  std::unordered_map<Token, bool, TokenHasher> token_map;
  CollectionParserInfo parser_info;

  // The function defined below works as follows:
  // 1. Acquire lock for reading from docword file
  // 2. Read num_items_per_batch lines from docword file, and store them in a local buffer (vector<string>)
  // 3. Release the lock
  // 4. Parse strings, form a batch, and save it to disk
  // Steps 1-4 are repeated in a while loop until there is no content left in docword file.
  // Multiple copies of the function can work in parallel.
  auto func = [&docword, &global_line_no, &progress, &batch_name_generator, &lock,
               &parser_info, &token_map, config]() {
    while (true) {
      // The following variable remembers at which line the batch has started.
      // It helps to create informative error message (including line number)
      // if later the code discovers a problem when parsing the line.
      int first_line_no_for_batch = -1;

      std::vector<std::string> all_strs_for_batch;
      std::string batch_name;
      BatchCollector batch_collector;

      {
        std::lock_guard<std::mutex> guard(lock);
        first_line_no_for_batch = global_line_no;
        if (docword.eof()) {
          return;
        }

        while ((int64_t) all_strs_for_batch.size() < config.num_items_per_batch()) {
          std::string str;
          std::getline(docword, str);
          global_line_no++;
          progress.Set(docword.tellg());
          if (docword.eof()) {
            break;
          }

          all_strs_for_batch.push_back(str);
        }

        if (all_strs_for_batch.size() > 0) {
          batch_name = batch_name_generator.next_name(batch_collector.batch());
        }
      }

      for (int str_index = 0; str_index < (int64_t) all_strs_for_batch.size(); ++str_index) {
        std::string str = all_strs_for_batch[str_index];
        const int line_no = first_line_no_for_batch + str_index;

        std::vector<std::string> strs;
        boost::split(strs, str, boost::is_any_of(" \t\r"));

        if (strs.size() <= 1) {
          std::stringstream ss;
          ss << "Error in " << config.docword_file_path() << ":" << line_no
             << " has too few entries: " << str;
          BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
        }

        std::string item_title = strs[0];

        ClassId class_id = DefaultClass;
        for (unsigned elem_index = 1; elem_index < strs.size(); ++elem_index) {
          std::string elem = strs[elem_index];
          if (elem.size() == 0) {
            continue;
          }

          if (elem[0] == '|') {
            class_id = elem.substr(1);
            if (class_id.empty()) {
              class_id = DefaultClass;
            }
            continue;
          }

          // Skip token when it is not among modalities that user has requested to parse
          if (!useClassId(class_id, config)) {
            continue;
          }

          float token_weight = 1.0f;
          std::string token = elem;
          size_t split_index = elem.find(':');
          if (split_index != std::string::npos) {
            if (split_index == 0 || split_index == (elem.size() - 1)) {
              std::stringstream ss;
              ss << "Error in " << config.docword_file_path() << ":" << line_no
                 << ", entries can not start or end with colon: " << elem;
              BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
            }
            token = elem.substr(0, split_index);
            std::string token_occurences_string = elem.substr(split_index + 1);
            try {
              token_weight = boost::lexical_cast<float>(token_occurences_string);
            }
            catch (boost::bad_lexical_cast &) {
              std::stringstream ss;
              ss << "Error in " << config.docword_file_path() << ":" << line_no
                 << ", can not parse integer number of occurences: " << elem;
              BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
            }
          }

          batch_collector.Record(artm::core::Token(class_id, token), token_weight);
        }

        batch_collector.FinishItem(line_no, item_title);
      }

      if (all_strs_for_batch.size() > 0) {
        artm::Batch batch;
        {
          std::lock_guard<std::mutex> guard(lock);
          batch = batch_collector.FinishBatch(&parser_info);
          for (int token_id = 0; token_id < batch.token_size(); ++token_id) {
            token_map[artm::core::Token(batch.class_id(token_id), batch.token(token_id))] = true;
          }
        }
        ::artm::core::Helpers::SaveBatch(batch, config.target_folder(), batch_name);
      }
    }
  };

  int num_threads = config.num_threads();
  if (!config.has_num_threads() || config.num_threads() < 0) {
    unsigned int n = std::thread::hardware_concurrency();
    if (n == 0) {
      LOG(INFO) << "CollectionParserConfig.num_threads is set to 1 (default)";
      num_threads = 1;
    } else {
      LOG(INFO) << "CollectionParserConfig.num_threads is automatically set to " << n;
      num_threads = n;
    }
  }

  Helpers::CreateFolderIfNotExists(config.target_folder());

  // The func may throw an exception if docword is malformed.
  // This exception will be re-thrown on the main thread.
  // http://stackoverflow.com/questions/14222899/exception-propagation-and-stdfuture
  std::vector<std::shared_future<void>> tasks;
  for (int i = 0; i < num_threads; i++) {
    tasks.push_back(std::move(std::async(std::launch::async, func)));
  }

  for (int i = 0; i < num_threads; i++) {
    tasks[i].get();
  }

  parser_info.set_dictionary_size(token_map.size());
  return parser_info;
}

CollectionParserInfo CollectionParser::Parse() {
  TokenMap token_map;
  switch (config_.format()) {
    case CollectionParserConfig_CollectionFormat_BagOfWordsUci:
      token_map = ParseVocabBagOfWordsUci();
      return ParseDocwordBagOfWordsUci(&token_map);

    case CollectionParserConfig_CollectionFormat_MatrixMarket:
      token_map = ParseVocabMatrixMarket();
      return ParseDocwordBagOfWordsUci(&token_map);

    case CollectionParserConfig_CollectionFormat_VowpalWabbit:
      return ParseVowpalWabbit();

    default:
      BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
        "CollectionParserConfig.format", config_.format()));
  }
}

}  // namespace core
}  // namespace artm
// vim: set ts=2 sw=2:
