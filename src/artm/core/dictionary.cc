// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/dictionary.h"

#include <fstream>
#include <string>
#include <map>
#include <memory>

#include "boost/algorithm/string.hpp"
#include "boost/algorithm/string/predicate.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

#include "artm/core/helpers.h"
#include "artm/utility/ifstream_or_cin.h"

using ::artm::utility::ifstream_or_cin;

namespace artm {
namespace core {

Dictionary::Dictionary(const artm::DictionaryData& data) {
  if (data.cooc_value_size() == 0) {
    for (int index = 0; index < data.token_size(); ++index) {
      ClassId class_id = data.class_id_size() ? data.class_id(index) : DefaultClass;
      bool has_token_value = data.token_value_size() > 0;
      bool has_token_tf = data.token_tf_size() > 0;
      bool has_token_df = data.token_df_size() > 0;
      entries_.push_back(DictionaryEntry(Token(class_id, data.token(index)),
        has_token_value ? data.token_value(index) : 0.0f,
        has_token_tf ? data.token_tf(index): 0.0f,
        has_token_df ? data.token_df(index): 0.0f));

      token_index_.insert(std::make_pair(entries_[index].token(), index));
    }
  } else {
    LOG(ERROR) << "Can't create Dictionary using the cooc part of DictionaryData";
  }
}

std::vector<std::shared_ptr<artm::DictionaryData> >
Dictionary::ImportData(const ImportDictionaryArgs& args) {
  if (!boost::algorithm::ends_with(args.file_name(), ".dict"))
    BOOST_THROW_EXCEPTION(CorruptedMessageException("The importing dictionary should have .dict exstension, abort."));

  std::ifstream fin(args.file_name(), std::ifstream::binary);
  if (!fin.is_open())
    BOOST_THROW_EXCEPTION(DiskReadException("Unable to open file " + args.file_name()));

  LOG(INFO) << "Importing dictionary " << args.dictionary_name() << " from " << args.file_name();

  char version;
  fin >> version;
  if (version != 0) {
    std::stringstream ss;
    ss << "Unsupported fromat version: " << static_cast<int>(version);
    BOOST_THROW_EXCEPTION(DiskReadException(ss.str()));
  }

  std::vector<std::shared_ptr<artm::DictionaryData> > import_data;
  while (!fin.eof()) {
    int length;
    fin.read(reinterpret_cast<char *>(&length), sizeof(length));
    if (fin.eof())
      break;

    if (length <= 0)
      BOOST_THROW_EXCEPTION(CorruptedMessageException("Unable to read from " + args.file_name()));

    std::string buffer(length, '\0');
    fin.read(&buffer[0], length);
    ::artm::DictionaryData dict_data;
    if (!dict_data.ParseFromArray(buffer.c_str(), length))
      BOOST_THROW_EXCEPTION(CorruptedMessageException("Unable to read from " + args.file_name()));

    import_data.push_back(std::make_shared<artm::DictionaryData>(dict_data));
  }

  fin.close();

  return import_data;
}

void Dictionary::Export(const ExportDictionaryArgs& args,
                            ThreadSafeDictionaryCollection* dictionaries) {
  std::string file_name = args.file_name();
  if (!boost::algorithm::ends_with(file_name, ".dict")) {
    LOG(WARNING) << "The exporting dictionary should have .dict extension, it will be added to file name";
    file_name += ".dict";
  }

  if (boost::filesystem::exists(file_name))
    BOOST_THROW_EXCEPTION(DiskWriteException("File already exists: " + file_name));

  std::ofstream fout(file_name, std::ofstream::binary);
  if (!fout.is_open())
    BOOST_THROW_EXCEPTION(DiskReadException("Unable to create file " + file_name));

  std::shared_ptr<Dictionary> dict_ptr = dictionaries->get(args.dictionary_name());
  if (dict_ptr == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation("Dictionary " +
        args.dictionary_name() + " does not exist or has no tokens"));

  LOG(INFO) << "Exporting dictionary " << args.dictionary_name() << " to " << file_name;

  const int token_size = static_cast<int>(dict_ptr->size());

  // ToDo: MelLain
  // Add ability to save and load several token_dict_data
  // int tokens_per_chunk = std::min<int>(token_size, 3e+7);

  const char version = 0;
  fout << version;

  DictionaryData token_dict_data;
  token_dict_data.set_name(args.dictionary_name());
  for (int token_id = 0; token_id < token_size; ++token_id) {
    auto entry = dict_ptr->entry(token_id);
    token_dict_data.add_token(entry->token().keyword);
    token_dict_data.add_class_id(entry->token().class_id);
    token_dict_data.add_token_value(entry->token_value());
    token_dict_data.add_token_tf(entry->token_tf());
    token_dict_data.add_token_df(entry->token_df());
  }
  std::string str = token_dict_data.SerializeAsString();
  int length = static_cast<int>(str.size());
  fout.write(reinterpret_cast<char *>(&length), sizeof(length));
  fout << str;

  DictionaryData cooc_dict_data;
  int current_cooc_length = 0;
  const int max_cooc_length = 10 * 1000 * 1000;
  if (dict_ptr->cooc_values().size()) {
    for (int token_id = 0; token_id < token_size; ++token_id) {
      auto entry = dict_ptr->entry(token_id);
      auto cooc_info = dict_ptr->cooc_info(entry->token());

      if (cooc_info != nullptr) {
        for (auto iter = cooc_info->begin(); iter != cooc_info->end(); ++iter) {
          cooc_dict_data.add_cooc_first_index(token_id);
          cooc_dict_data.add_cooc_second_index(iter->first);
          cooc_dict_data.add_cooc_value(iter->second);
          current_cooc_length++;
        }
      }

      if ((current_cooc_length >= max_cooc_length) || ((token_id + 1) == token_size)) {
        std::string str = cooc_dict_data.SerializeAsString();
        int length = static_cast<int>(str.size());
        fout.write(reinterpret_cast<char *>(&length), sizeof(length));
        fout << str;
        cooc_dict_data.clear_cooc_first_index();
        cooc_dict_data.clear_cooc_second_index();
        cooc_dict_data.clear_cooc_value();
        current_cooc_length = 0;
      }
    }
  }

  fout.close();
  LOG(INFO) << "Export completed, token_size = " << dict_ptr->size();
}

// TokenValues is used only from Dictionary::Gather method
class TokenValues {
 public:
  TokenValues() : token_value(0.0f), token_tf(0.0f), token_df(0.0f) { }

  float token_value;
  float token_tf;
  float token_df;
};

std::pair<std::shared_ptr<DictionaryData>, std::shared_ptr<DictionaryData> >
Dictionary::Gather(const GatherDictionaryArgs& args,
                   const ThreadSafeCollectionHolder<std::string, Batch>& mem_batches) {
  std::unordered_map<Token, TokenValues, TokenHasher> token_freq_map;
  std::vector<std::string> batches;

  if (args.has_data_path()) {
    for (auto& batch_path : Helpers::ListAllBatches(args.data_path()))
      batches.push_back(batch_path.string());
    LOG(INFO) << "Found " << batches.size() << " batches in '" << args.data_path() << "' folder";
  } else {
    for (auto& batch : args.batch_path())
      batches.push_back(batch);
  }

  int total_items_count = 0;
  double sum_w_tf = 0.0;
  for (const std::string& batch_file : batches) {
    std::shared_ptr<Batch> batch_ptr = mem_batches.get(batch_file);
    try {
      if (batch_ptr == nullptr) {
        batch_ptr = std::make_shared<Batch>();
        ::artm::core::Helpers::LoadMessage(batch_file, batch_ptr.get());
      }
    } catch(std::exception& ex) {
        LOG(ERROR) << ex.what() << ", the batch will be skipped.";
        continue;
    }

    if (batch_ptr->token_size() == 0)
      BOOST_THROW_EXCEPTION(InvalidOperation(
        "Dictionary::Gather() can not process batches with empty Batch.token field."));

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
      for (int i = 0; i < batch.token_size(); ++i)
        token_df[i] += local_token_df[i] ? 1.0 : 0.0;
    }

    for (int index = 0; index < batch.token_size(); ++index) {
      // unordered_map.operator[] creates element using default constructor if the key doesn't exist
      TokenValues& token_info = token_freq_map[Token(batch.class_id(index), batch.token(index))];
      token_info.token_tf += token_n_w[index];
      sum_w_tf += token_n_w[index];
      token_info.token_df += token_df[index];
    }
  }

  for (auto iter = token_freq_map.begin(); iter != token_freq_map.end(); ++iter)
    iter->second.token_value = static_cast<float>(iter->second.token_tf / sum_w_tf);

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
        if (vocab.eof())
          break;

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
    } catch(std::exception& ex) {
      use_vocab_file = false;
      LOG(ERROR) << ex.what() << ", dictionary will be gathered in random token order";
    }
  }

  if (!use_vocab_file) {  // fill dictionary in map order
    collection_vocab.clear();
    for (auto iter = token_freq_map.begin(); iter != token_freq_map.end(); ++iter)
      collection_vocab.push_back(iter->first);
  }

  auto dictionary_data = std::make_shared<artm::DictionaryData>();
  dictionary_data->set_name(args.dictionary_target_name());

  for (auto& token : collection_vocab) {
    dictionary_data->add_token(token.keyword);
    dictionary_data->add_class_id(token.class_id);
    dictionary_data->add_token_tf(token_freq_map[token].token_tf);
    dictionary_data->add_token_df(token_freq_map[token].token_df);
    dictionary_data->add_token_value(token_freq_map[token].token_value);
  }

  // parse the cooc info and append it to DictionaryData
  auto cooc_dictionary_data = std::make_shared<artm::DictionaryData>();
  cooc_dictionary_data->set_name(args.dictionary_target_name());

  if (args.has_cooc_file_path()) {
    try {
      ifstream_or_cin stream_or_cin(args.cooc_file_path());
      std::istream& user_cooc_data = stream_or_cin.get_stream();

      // Craft the co-occurence part of dictionary
      int index = 0;
      std::string str;
      bool last_line = false;
      while (!user_cooc_data.eof()) {
        if (last_line) {
          std::stringstream ss;
          ss << "Empty pair of tokens at line " << index << ", file " << args.cooc_file_path();
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
          ss << "Error at line " << index << ", file " << args.cooc_file_path()
             << ". Expected format: <token_id_1> <token_id_2> {<cooc_value>}";
          BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
        }

        if (strs.size() != 3) {
          std::stringstream ss;
          ss << "Error at line " << index << ", file " << args.cooc_file_path()
             << ". Number of values in all lines should be equal to 3";
          BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
        }

        int first_index = std::stoi(strs[0]);
        int second_index = std::stoi(strs[1]);
        float value = std::stof(strs[2]);

        cooc_dictionary_data->add_cooc_first_index(first_index);
        cooc_dictionary_data->add_cooc_second_index(second_index);
        cooc_dictionary_data->add_cooc_value(value);

        if (args.symmetric_cooc_values()) {
          cooc_dictionary_data->add_cooc_first_index(second_index);
          cooc_dictionary_data->add_cooc_second_index(first_index);
          cooc_dictionary_data->add_cooc_value(value);
        }
      }
    } catch(std::exception& ex) {
      cooc_dictionary_data->clear_cooc_first_index();
      cooc_dictionary_data->clear_cooc_second_index();
      cooc_dictionary_data->clear_cooc_value();
      LOG(ERROR) << ex.what() << ", dictionary will be gathered without cooc info";
    }
  }

  return std::pair<std::shared_ptr<DictionaryData>,
                   std::shared_ptr<DictionaryData> >(dictionary_data, cooc_dictionary_data);
}

std::pair<std::shared_ptr<DictionaryData>, std::shared_ptr<DictionaryData> >
Dictionary::Filter(const FilterDictionaryArgs& args, ThreadSafeDictionaryCollection* dictionaries) {
  auto src_dictionary_ptr = dictionaries->get(args.dictionary_name());
  if (src_dictionary_ptr == nullptr) {
    LOG(ERROR) << "Dictionary::Filter(): filter was requested for non-exists dictionary '"
      << args.dictionary_name() << "', operation was aborted";
  }

  auto dictionary_data = std::make_shared<artm::DictionaryData>();
  dictionary_data->set_name(args.dictionary_target_name());

  auto& src_entries = src_dictionary_ptr->entries();
  auto& dictionary_token_index = src_dictionary_ptr->token_index();
  std::unordered_map<int, int> old_index_new_index;

  float size = static_cast<float>(src_dictionary_ptr->size());

  int accepted_tokens_count = 0;
  for (auto& entry : src_entries) {
    if (!args.has_class_id() || (entry.token().class_id == args.class_id())) {
      if (args.has_min_df() && entry.token_df() < args.min_df()) continue;
      if (args.has_max_df() && entry.token_df() >= args.max_df()) continue;

      if (args.has_min_df_rate() && entry.token_df() < (args.min_df_rate() * size)) continue;
      if (args.has_max_df_rate() && entry.token_df() >= (args.max_df_rate() * size)) continue;

      if (args.has_min_tf() && entry.token_tf() < args.min_tf()) continue;
      if (args.has_max_tf() && entry.token_tf() >= args.max_tf()) continue;
    }

    // all filters were passed, add token to the new dictionary
    Token token = entry.token();
    accepted_tokens_count += 1;
    dictionary_data->add_token(token.keyword);
    dictionary_data->add_class_id(token.class_id);
    dictionary_data->add_token_df(entry.token_df());
    dictionary_data->add_token_tf(entry.token_tf());
    dictionary_data->add_token_value(entry.token_value());

    old_index_new_index.insert(std::pair<int, int>(dictionary_token_index.find(token)->second,
                                                   accepted_tokens_count - 1));
  }

  auto cooc_dictionary_data = std::make_shared<artm::DictionaryData>();
  auto& cooc_values = src_dictionary_ptr->cooc_values();

  for (auto iter = cooc_values.begin(); iter != cooc_values.end(); ++iter) {
    auto first_index_iter = old_index_new_index.find(iter->first);
    if (first_index_iter == old_index_new_index.end()) continue;

    for (auto cooc_iter = iter->second.begin(); cooc_iter != iter->second.end(); ++cooc_iter) {
      auto second_index_iter = old_index_new_index.find(cooc_iter->first);
      if (second_index_iter == old_index_new_index.end()) continue;

      cooc_dictionary_data->add_cooc_first_index(first_index_iter->second);
      cooc_dictionary_data->add_cooc_second_index(second_index_iter->second);
      cooc_dictionary_data->add_cooc_value(cooc_iter->second);
    }
  }

  return std::pair<std::shared_ptr<DictionaryData>,
                   std::shared_ptr<DictionaryData> >(dictionary_data, cooc_dictionary_data);
}

void Dictionary::Append(const DictionaryData& data) {
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
  int k = static_cast<int>(tokens_to_score.size());
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
