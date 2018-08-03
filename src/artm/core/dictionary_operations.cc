// Copyright 2017, Additive Regularization of Topic Models.

#include <algorithm>
#include <climits>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <vector>
#include <utility>

#include "boost/algorithm/string.hpp"
#include "boost/algorithm/string/predicate.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

#include "artm/core/helpers.h"
#include "artm/utility/ifstream_or_cin.h"

#include "artm/core/dictionary_operations.h"

using ::artm::utility::ifstream_or_cin;

namespace artm {
namespace core {

std::shared_ptr<Dictionary> DictionaryOperations::Create(const DictionaryData& data) {
  auto dictionary = std::make_shared<Dictionary>(Dictionary(data.name()));

  if (data.cooc_value_size() == 0) {
    dictionary->SetNumItems(data.num_items_in_collection());
    for (int index = 0; index < data.token_size(); ++index) {
      ClassId class_id = data.class_id_size() ? data.class_id(index) : DefaultClass;
      bool has_token_value = data.token_value_size() > 0;
      bool has_token_tf = data.token_tf_size() > 0;
      bool has_token_df = data.token_df_size() > 0;
      dictionary->AddEntry(DictionaryEntry(Token(class_id, data.token(index)),
        has_token_value ? data.token_value(index) : 0.0f,
        has_token_tf ? data.token_tf(index) : 0.0f,
        has_token_df ? data.token_df(index) : 0.0f));
    }
  } else {
    LOG(ERROR) << "Can't create Dictionary using the cooc part of DictionaryData";
  }

  return dictionary;
}

void DictionaryOperations::Export(const ExportDictionaryArgs& args, const Dictionary& dict) {
  std::string file_name = args.file_name();
  if (!boost::algorithm::ends_with(file_name, ".dict")) {
    LOG(WARNING) << "The exporting dictionary should have .dict extension, it will be added to file name";
    file_name += ".dict";
  }

  if (boost::filesystem::exists(file_name)) {
    BOOST_THROW_EXCEPTION(DiskWriteException("File already exists: " + file_name));
  }

  std::ofstream fout(file_name, std::ofstream::binary);
  if (!fout.is_open()) {
    BOOST_THROW_EXCEPTION(DiskReadException("Unable to create file " + file_name));
  }

  if (!dict.has_valid_cooc_state()) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Dictionary " +
      args.dictionary_name() + " has invalid cooc state (num values: " +
      std::to_string(dict.cooc_values().size()) +
      ", num tfs: " + std::to_string(dict.cooc_tfs().size()) +
      ", num dfs: " + std::to_string(dict.cooc_dfs().size()) + ")"));
  }

  LOG(INFO) << "Exporting dictionary " << args.dictionary_name() << " to " << file_name;

  const int token_size = static_cast<int>(dict.size());

  // ToDo: MelLain
  // Add ability to save and load several token_dict_data
  // int tokens_per_chunk = std::min<int>(token_size, 3e+7);

  const char version = 0;
  fout << version;

  DictionaryData token_dict_data;
  token_dict_data.set_name(args.dictionary_name());
  token_dict_data.set_num_items_in_collection(dict.num_items());
  for (int token_id = 0; token_id < token_size; ++token_id) {
    auto entry = dict.entry(token_id);
    token_dict_data.add_token(entry->token().keyword);
    token_dict_data.add_class_id(entry->token().class_id);
    token_dict_data.add_token_value(entry->token_value());
    token_dict_data.add_token_tf(entry->token_tf());
    token_dict_data.add_token_df(entry->token_df());
  }

  std::string str = token_dict_data.SerializeAsString();
  if (str.size() >= kProtobufCodedStreamTotalBytesLimit) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Dictionary " +
      args.dictionary_name() + " is too large to export"));
  }

  int length = static_cast<int>(str.size());
  fout.write(reinterpret_cast<char *>(&length), sizeof(length));
  fout << str;

  DictionaryData cooc_dict_data;
  int current_cooc_length = 0;
  const int max_cooc_length = 10 * 1000 * 1000;
  if (dict.cooc_values().size()) {
    for (int token_id = 0; token_id < token_size; ++token_id) {
      auto entry = dict.entry(token_id);
      auto cooc_values_info = dict.token_cooc_values(entry->token());
      auto cooc_tfs_info = dict.token_cooc_tfs(entry->token());
      auto cooc_dfs_info = dict.token_cooc_dfs(entry->token());

      if (cooc_values_info != nullptr) {
        for (auto iter = cooc_values_info->begin(); iter != cooc_values_info->end(); ++iter) {
          cooc_dict_data.add_cooc_first_index(token_id);
          cooc_dict_data.add_cooc_second_index(iter->first);
          cooc_dict_data.add_cooc_value(iter->second);
          if (cooc_tfs_info != nullptr) {
            auto tf_iter = cooc_tfs_info->find(iter->first);
            auto df_iter = cooc_dfs_info->find(iter->first);

            if (tf_iter == cooc_tfs_info->end() || df_iter == cooc_dfs_info->end()) {
              BOOST_THROW_EXCEPTION(InvalidOperation("Dictionary " +
                  args.dictionary_name() + " has internal cooc tf/df inconsistence"));
            }

            cooc_dict_data.add_cooc_tf(tf_iter->second);
            cooc_dict_data.add_cooc_df(df_iter->second);
          }
          current_cooc_length++;
        }
      }

      if ((current_cooc_length >= max_cooc_length) || ((token_id + 1) == token_size)) {
        std::string str = cooc_dict_data.SerializeAsString();
        if (str.size() >= kProtobufCodedStreamTotalBytesLimit) {
          BOOST_THROW_EXCEPTION(InvalidOperation(
            "Unable to serialize coocurence information in Dictionary " +
            args.dictionary_name()));
        }

        int length = static_cast<int>(str.size());
        fout.write(reinterpret_cast<char *>(&length), sizeof(length));
        fout << str;
        cooc_dict_data.clear_cooc_first_index();
        cooc_dict_data.clear_cooc_second_index();
        cooc_dict_data.clear_cooc_value();
        cooc_dict_data.clear_cooc_tf();
        cooc_dict_data.clear_cooc_df();
        current_cooc_length = 0;
      }
    }
  }

  fout.close();
  LOG(INFO) << "Export completed, token_size = " << dict.size();
}

std::shared_ptr<Dictionary> DictionaryOperations::Import(const ImportDictionaryArgs& args) {
  auto dictionary = std::make_shared<Dictionary>(Dictionary(args.dictionary_name()));

  if (!boost::algorithm::ends_with(args.file_name(), ".dict")) {
    BOOST_THROW_EXCEPTION(CorruptedMessageException(
      "The importing dictionary should have .dict exstension, abort."));
  }

  std::ifstream fin(args.file_name(), std::ifstream::binary);
  if (!fin.is_open()) {
    BOOST_THROW_EXCEPTION(DiskReadException("Unable to open file " + args.file_name()));
  }

  LOG(INFO) << "Importing dictionary " << args.dictionary_name() << " from " << args.file_name();

  char version;
  fin >> version;
  if (version != 0) {
    std::stringstream ss;
    ss << "Unsupported format version: " << static_cast<int>(version);
    BOOST_THROW_EXCEPTION(DiskReadException(ss.str()));
  }

  while (!fin.eof()) {
    int length;
    fin.read(reinterpret_cast<char *>(&length), sizeof(length));
    if (fin.eof()) {
      break;
    }

    if (length <= 0) {
      BOOST_THROW_EXCEPTION(CorruptedMessageException("Unable to read from " + args.file_name()));
    }

    std::string buffer(length, '\0');
    fin.read(&buffer[0], length);
    ::artm::DictionaryData dict_data;
    if (!dict_data.ParseFromArray(buffer.c_str(), length)) {
      BOOST_THROW_EXCEPTION(CorruptedMessageException("Unable to read from " + args.file_name()));
    }

    // move data from DictionaryData into dictionary
    if ((dict_data.token_size() > 0) == (dict_data.cooc_value_size() > 0)) {
      BOOST_THROW_EXCEPTION(CorruptedMessageException("Error while reading from " + args.file_name()));
    }

    // part with main dictionary
    if (dict_data.token_size() > 0) {
      dictionary->SetNumItems(dict_data.num_items_in_collection());
      for (int token_id = 0; token_id < dict_data.token_size(); ++token_id) {
        dictionary->AddEntry(DictionaryEntry(
          Token(dict_data.class_id(token_id), dict_data.token(token_id)),
          dict_data.token_value(token_id), dict_data.token_tf(token_id), dict_data.token_df(token_id)));
      }
    }

    // part with cooc dictionary
    if (dict_data.cooc_value_size() > 0) {
      for (int index = 0; index < dict_data.cooc_first_index_size(); ++index) {
        int index_1 = dict_data.cooc_first_index(index);
        int index_2 = dict_data.cooc_second_index(index);
        dictionary->AddCoocValue(index_1, index_2, dict_data.cooc_value(index));

        if (dict_data.cooc_tf_size() > 0) {
          dictionary->AddCoocTf(index_1, index_2, dict_data.cooc_tf(index));
          dictionary->AddCoocDf(index_1, index_2, dict_data.cooc_df(index));
        }
      }
    }
  }
  fin.close();

  return dictionary;
}

// TokenValues is used only from Dictionary::Gather method
class TokenValues {
 public:
  TokenValues()
      : token_value(0.0f)
      , token_tf(0.0f)
      , token_df(0.0f) { }

  float token_value;
  float token_tf;
  float token_df;
};

std::shared_ptr<Dictionary> DictionaryOperations::Gather(const GatherDictionaryArgs& args,
  const ThreadSafeCollectionHolder<std::string, Batch>& mem_batches) {
  auto dictionary = std::make_shared<Dictionary>(Dictionary(args.dictionary_target_name()));

  std::unordered_map<Token, TokenValues, TokenHasher> token_freq_map;
  std::vector<std::string> batches;

  if (args.has_data_path()) {
    for (const auto& batch_path : Helpers::ListAllBatches(args.data_path())) {
      batches.push_back(batch_path.string());
    }
    LOG(INFO) << "Found " << batches.size() << " batches in '" << args.data_path() << "' folder";
  } else {
    for (const auto& batch : args.batch_path()) {
      batches.push_back(batch);
    }
  }

  int total_items_count = 0;
  std::unordered_map<ClassId, float> sum_w_tf;
  for (const std::string& batch_file : batches) {
    std::shared_ptr<Batch> batch_ptr = mem_batches.get(batch_file);
    try {
      if (batch_ptr == nullptr) {
        batch_ptr = std::make_shared<Batch>();
        ::artm::core::Helpers::LoadMessage(batch_file, batch_ptr.get());
      }
    }
    catch (std::exception& ex) {
      LOG(ERROR) << ex.what() << ", the batch will be skipped.";
      continue;
    }

    if (batch_ptr->token_size() == 0) {
      BOOST_THROW_EXCEPTION(InvalidOperation(
      "Dictionary::Gather() can not process batches with empty Batch.token field."));
    }

    const Batch& batch = *batch_ptr;
    std::vector<float> token_df(batch.token_size(), 0);
    std::vector<float> token_n_w(batch.token_size(), 0);

    for (int item_id = 0; item_id < batch.item_size(); ++item_id) {
      total_items_count++;
      // Find cumulative weight for each token in item
      // (assume that token might have multiple occurence in each item)
      std::vector<bool> local_token_df(batch.token_size(), false);
      const Item& item = batch.item(item_id);

      for (int token_index = 0; token_index < item.token_weight_size(); ++token_index) {
        const float token_weight = item.token_weight(token_index);
        const int token_id = item.token_id(token_index);
        token_n_w[token_id] += token_weight;
        local_token_df[token_id] = true;
      }

      for (int i = 0; i < batch.token_size(); ++i) {
        token_df[i] += local_token_df[i] ? 1.0f : 0.0f;
      }
    }

    for (int index = 0; index < batch.token_size(); ++index) {
      // unordered_map.operator[] creates element using default constructor if the key doesn't exist
      const ClassId& token_class_id = batch.class_id(index);
      TokenValues& token_info = token_freq_map[Token(token_class_id, batch.token(index))];
      token_info.token_tf += token_n_w[index];
      token_info.token_df += token_df[index];

      sum_w_tf[token_class_id] += token_n_w[index];
    }
  }

  for (auto& token_freq : token_freq_map) {
    token_freq.second.token_value = static_cast<float>(token_freq.second.token_tf /
                                                       sum_w_tf[token_freq.first.class_id]);
  }

  LOG(INFO) << "Find " << token_freq_map.size()
    << " unique tokens in " << total_items_count << " items";

  // create DictionaryDataMessage using token_freq_map and vocab file
  // if vocab file is given
  std::vector<Token> collection_vocab;
  std::unordered_map<Token, int, TokenHasher> token_to_token_id;
  bool use_vocab_file = args.has_vocab_file_path();

  if (use_vocab_file) {
    try {
      ifstream_or_cin stream_or_cin(args.vocab_file_path());
      std::istream& vocab = stream_or_cin.get_stream();

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
          ss << "Empty token at line " << (token_id + 1) << ", file " << args.vocab_file_path();
          BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
        }

        std::vector<std::string> strs;
        boost::split(strs, str, boost::is_any_of("\t "));
        if ((strs.size() == 0) || (strs.size() > 2)) {
          std::stringstream ss;
          ss << "Error at line " << (token_id + 1) << ", file " << args.vocab_file_path()
            << ". Expected format: <token> [<class_id>]";
          BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
        }

        ClassId class_id = (strs.size() == 2) ? strs[1] : DefaultClass;
        Token token(class_id, strs[0]);

        if (token_to_token_id.find(token) != token_to_token_id.end()) {
          std::stringstream ss;
          ss << "Token (" << token.keyword << ", " << token.class_id << "' found twice, lines "
            << (token_to_token_id.find(token)->second + 1)
            << " and " << (token_id + 1) << ", file " << args.vocab_file_path();
          BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
        }

        collection_vocab.push_back(token);
        token_to_token_id.insert(std::make_pair(token, token_id));
        token_id++;
      }
    }
    catch (std::exception& ex) {
      use_vocab_file = false;
      LOG(ERROR) << ex.what() << ", dictionary will be gathered in random token order";
    }
  }

  if (!use_vocab_file) {  // fill dictionary in map order
    collection_vocab.clear();
    for (auto iter = token_freq_map.begin(); iter != token_freq_map.end(); ++iter) {
      collection_vocab.push_back(iter->first);
    }
  }

  dictionary->SetNumItems(total_items_count);
  for (const auto& token : collection_vocab) {
    dictionary->AddEntry(DictionaryEntry(token, token_freq_map[token].token_value,
      token_freq_map[token].token_tf, token_freq_map[token].token_df));
  }

  if (args.has_cooc_file_path()) {
    try {
      ifstream_or_cin stream_or_cin(args.cooc_file_path());
      std::istream& user_cooc_data = stream_or_cin.get_stream();

      // Craft the co-occurence part of dictionary
      std::string str;
      while (!user_cooc_data.eof()) {
        std::getline(user_cooc_data, str);
        boost::algorithm::trim(str);

        ClassId first_token_class_id = DefaultClass;  // Here's how modality is indicated in output file
        std::vector<std::string> strs;
        boost::split(strs, str, boost::is_any_of(" :\t\r"));
        unsigned pos_of_first_token = 0;
        // Find modality and position of the first token
        for (; pos_of_first_token < strs.size() && (strs[pos_of_first_token].empty() ||
                                                    strs[pos_of_first_token][0] == '|'); ++pos_of_first_token) {
          if (!strs[pos_of_first_token].empty()) {
            first_token_class_id = strs[pos_of_first_token];
            first_token_class_id.erase(0);
          }
        }
        if (pos_of_first_token >= strs.size()) {
          continue;
        }
        std::string first_token_str = strs[pos_of_first_token];
        Token first_token(first_token_class_id, first_token_str);
        auto first_token_ptr = token_to_token_id.find(first_token);
        if (first_token_ptr == token_to_token_id.end()) {
          std::stringstream ss;
          ss << "Token (" << first_token.keyword << ", " << first_token.class_id << ") not found in vocab";
          BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
        }
        unsigned not_a_word_counter = 0;
        for (unsigned i = pos_of_first_token + 1; i + not_a_word_counter < strs.size(); i += 2) {
          ClassId second_token_class_id = first_token_class_id;
          for (; i + not_a_word_counter < strs.size() && (strs[i + not_a_word_counter].empty() ||
                                                          strs[i + not_a_word_counter][0] == '|');
                                                          ++not_a_word_counter) {
            if (!strs[i + not_a_word_counter].empty()) {
              second_token_class_id = strs[i + not_a_word_counter];
              second_token_class_id.erase(0);
            }
          }
          if (i + not_a_word_counter + 1 >= strs.size()) {
            break;
          }
          std::string second_token_str = strs[i + not_a_word_counter];
          Token second_token(second_token_class_id, second_token_str);
          auto second_token_ptr = token_to_token_id.find(second_token);
          if (second_token_ptr == token_to_token_id.end()) {
            std::stringstream ss;
            ss << "Token (" << second_token.keyword << ", " << second_token.class_id << ") not found in vocab";
            BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
          }
          int first_index = first_token_ptr->second;
          int second_index = second_token_ptr->second;
          float value = std::stof(strs[i + not_a_word_counter + 1]);

          dictionary->AddCoocValue(first_index, second_index, value);

          // ToDo(MelLain): support adding tf/df in future

          if (args.symmetric_cooc_values()) {
            dictionary->AddCoocValue(second_index, first_index, value);
          }
        }
      }
    }
    catch (std::exception& ex) {
      dictionary->clear_cooc();
      LOG(ERROR) << ex.what() << ", dictionary will be gathered without cooc info";
    }
  }

  return dictionary;
}

std::shared_ptr<Dictionary> DictionaryOperations::Filter(const FilterDictionaryArgs& args, const Dictionary& dict) {
  auto dictionary = std::make_shared<Dictionary>(Dictionary(args.dictionary_target_name()));
  dictionary->SetNumItems(dict.num_items());

  auto& src_entries = dict.entries();
  auto& dictionary_token_index = dict.token_index();
  std::unordered_map<int, int> old_index_new_index;

  float size = static_cast<float>(dict.num_items());
  std::vector<bool> entries_mask(src_entries.size(), false);
  std::vector<float> df_values;
  double new_tf_normalizer = 0.0;

  for (int entry_index = 0; entry_index < (int64_t) src_entries.size(); entry_index++) {
    auto& entry = src_entries[entry_index];
    if (!args.has_class_id() || (entry.token().class_id == args.class_id())) {
      if (args.has_min_df() && entry.token_df() < args.min_df()) {
        continue;
      }

      if (args.has_max_df() && entry.token_df() >= args.max_df()) {
        continue;
      }

      if (args.has_min_df_rate() && entry.token_df() < (args.min_df_rate() * size)) {
        continue;
      }

      if (args.has_max_df_rate() && entry.token_df() >= (args.max_df_rate() * size)) {
        continue;
      }

      if (args.has_min_tf() && entry.token_tf() < args.min_tf()) {
        continue;
      }

      if (args.has_max_tf() && entry.token_tf() >= args.max_tf()) {
        continue;
      }
    }

    entries_mask[entry_index] = true;  // have passed all filters
    df_values.push_back(entry.token_df());
    new_tf_normalizer += entry.token_tf();
  }

  // Handle max_dictionary_size
  if (args.has_max_dictionary_size() && args.max_dictionary_size() < static_cast<int>(df_values.size())) {
    std::sort(df_values.begin(), df_values.end(), std::greater<float>());
    float min_df_due_to_size = df_values[args.max_dictionary_size()];

    for (int entry_index = 0; entry_index < (int64_t) src_entries.size();
            entry_index++) {
      auto& entry = src_entries[entry_index];
      if (entry.token_df() <= min_df_due_to_size) {
        entries_mask[entry_index] = false;
        new_tf_normalizer -= entry.token_tf();
      }
    }
  }

  int accepted_tokens_count = 0;
  for (int entry_index = 0; entry_index < (int64_t) src_entries.size(); entry_index++) {
    if (!entries_mask[entry_index]) {
      continue;
    }

    // all filters were passed, add token to the new dictionary
    auto& entry = src_entries[entry_index];
    ++accepted_tokens_count;
    if (args.recalculate_value()) {
      float value = static_cast<float>(new_tf_normalizer > 0.0 ? entry.token_tf() / new_tf_normalizer : 0.0);
      dictionary->AddEntry({ entry.token(), value, entry.token_tf(), entry.token_df() });
    } else {
      dictionary->AddEntry(entry);
    }

    old_index_new_index.insert(std::pair<int, int>(dictionary_token_index.find(entry.token())->second,
      accepted_tokens_count - 1));
  }

  auto& cooc_values = dict.cooc_values();

  for (auto iter = cooc_values.begin(); iter != cooc_values.end(); ++iter) {
    auto first_index_iter = old_index_new_index.find(iter->first);
    if (first_index_iter == old_index_new_index.end()) {
      continue;
    }

    for (auto cooc_iter = iter->second.begin(); cooc_iter != iter->second.end(); ++cooc_iter) {
      auto second_index_iter = old_index_new_index.find(cooc_iter->first);
      if (second_index_iter == old_index_new_index.end()) {
        continue;
      }

      dictionary->AddCoocValue(first_index_iter->second, second_index_iter->second, cooc_iter->second);
      // ToDo(MelLain): deal with tf/df
    }
  }

  return dictionary;
}

void DictionaryOperations::StoreIntoDictionaryData(const Dictionary& dict, DictionaryData* data) {
  data->set_name(dict.name());
  data->set_num_items_in_collection(dict.num_items());
  auto& entries = dict.entries();
  for (int i = 0; i < (int64_t) dict.size(); ++i) {
    data->add_token(entries[i].token().keyword);
    data->add_class_id(entries[i].token().class_id);
    data->add_token_value(entries[i].token_value());
    data->add_token_tf(entries[i].token_tf());
    data->add_token_df(entries[i].token_df());
  }
}

void DictionaryOperations::WriteDictionarySummaryToLog(const Dictionary& dict) {
  std::map<ClassId, int> entries_per_class;
  for (int i = 0; i < dict.size(); i++) {
    const DictionaryEntry* entry = dict.entry(i);
    if (entry != nullptr) {
      entries_per_class[entry->token().class_id]++;
    }
  }
  std::stringstream ss; ss << "Dictionary name='" << dict.name() << "' contains entries: ";
  for (auto const& x : entries_per_class) {
    ss << x.first << ":" << x.second << "; ";
  }
  LOG(INFO) << ss.str();
}

}  // namespace core
}  // namespace artm
