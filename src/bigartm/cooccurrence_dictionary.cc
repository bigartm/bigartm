// Copyright 2017, Additive Regularization of Topic Models.

#include "cooccurrence_dictionary.h"

#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <future>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <memory>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/utility.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

namespace fs = boost::filesystem;

CooccurrenceDictionary::CooccurrenceDictionary(const int window_width,
    const int cooc_min_tf, const int cooc_min_df,
    const std::string& path_to_vocab, const std::string& path_to_vw,
    const std::string& cooc_tf_file_path,
    const std::string& cooc_df_file_path,
    const std::string& ppmi_tf_file_path,
    const std::string& ppmi_df_file_path) : window_width_(window_width),
        cooc_min_tf_(cooc_min_tf), cooc_min_df_(cooc_min_df),
        path_to_vocab_(path_to_vocab), path_to_vw_(path_to_vw),
        cooc_tf_file_path_(cooc_tf_file_path),
        cooc_df_file_path_(cooc_df_file_path),
        ppmi_tf_file_path_(ppmi_tf_file_path),
        ppmi_df_file_path_(ppmi_df_file_path),
        total_num_of_pairs_(0), total_num_of_documents_(0) {
  // This function works as follows:
  // 1. Get content from a vocab file and put it in dictionary
  // 2. Read Vowpal Wabbit file by portions, calculate co-occurrences for
  // every portion and save it (cooccurrence batch) on external storage
  // 3. Read from external storage all the cooccurrence batches piece by
  // piece and create resulting file with all co-occurrences

  write_tf_cooc_ = cooc_tf_file_path_.size() != 0;
  write_df_cooc_ = cooc_df_file_path_.size() != 0;
  calculate_tf_ppmi_ = ppmi_tf_file_path_.size() != 0;
  calculate_df_ppmi_ = ppmi_df_file_path_.size() != 0;
  calculate_ppmi_ = calculate_tf_ppmi_ || calculate_df_ppmi_;
  calculate_tf_cooc_ = write_tf_cooc_ || calculate_tf_ppmi_;
  calculate_df_cooc_ = write_df_cooc_ || calculate_df_ppmi_;
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  fs::path dir(boost::lexical_cast<std::string>(uuid));
  if (fs::exists(dir))
    throw "Folder with uuid already exists";
  if (!fs::create_directory(dir))
    throw "Failed to create directory";
  path_to_batches_ = dir.string();
  open_files_counter_ = 0;
  max_num_of_open_files_ = 1000;
  num_of_threads_ = std::thread::hardware_concurrency();
  if (num_of_threads_ == 0)
    num_of_threads_ = 1;
  items_per_batch_ = SetItemsPerBatch();
  std::cout << "items per batch = " << items_per_batch_ << std::endl;
}

CooccurrenceDictionary::~CooccurrenceDictionary() {
  fs::remove_all(path_to_batches_);
}

void CooccurrenceDictionary::FetchVocab() {
  // This func reads words from vocab, sets them unique id and collects pair
  // in dictionary
  std::ifstream vocab(path_to_vocab_, std::ios::in);
  if (!vocab.is_open())
    throw "Failed to open vocab";
  int last_token_id = 1;
  std::string str;

  while (true) {
    getline(vocab, str);
    if (vocab.eof())
      break;
    boost::algorithm::trim(str);
    std::vector<std::string> strs;
    boost::split(strs, str, boost::is_any_of(" "));
    if (!strs[0].empty())
      if (strs.size() == 1 || strcmp(strs[1].c_str(), "@default_class") == 0)
        vocab_dictionary_.insert(std::make_pair(strs[0], last_token_id++)).second;
  }
}

int CooccurrenceDictionary::VocabDictionarySize() {
  return vocab_dictionary_.size();
}

void CooccurrenceDictionary::ReadVowpalWabbit() {
  // This func works as follows:
  // 1. Acquire lock for reading from vowpal wabbit file
  // 2. Read a portion (items_per_batch) of documents from file and save it
  // in a local buffer (vetor<string>)
  // 3. Release the lock
  // 4. Cut every document into words, search for them in dictionary and for
  // valid calculate co-occurrence, number of documents where these words
  // were found close enough (in a window with width window_width) and save
  // all in map (its own for each document)
  // 5. merge all maps into one
  // 6. If resulting map isn't empty create a batch and dump all information
  // on external storage
  // Repeat 1-6 for all portions (can work in parallel for different portions)

  std::ifstream vowpal_wabbit_doc(path_to_vw_, std::ios::in);
  if (!vowpal_wabbit_doc.is_open())
    throw "Failed to open vocab";
  std::mutex read_lock;

  auto func = [&]() {
    while (true) {
      std::vector<std::string> portion;

      {
        std::lock_guard<std::mutex> guard(read_lock);
        if (vowpal_wabbit_doc.eof())
          return;

        std::string str;
        while (static_cast<int>(portion.size()) < items_per_batch_) {
          getline(vowpal_wabbit_doc, str);
          if (vowpal_wabbit_doc.eof())
            break;
          portion.push_back(str);
        }
      }

      if (portion.size() == 0)
        continue;

      // First elem in external map is first_token_id, in internal it's
      // second_token_id
      CoocMap cooc_map;

      total_num_of_documents_ += static_cast<int>(portion.size());
      // When the document is processed (element of vector portion),
      // memory for it can be freed by calling pop_back() from vector
      // (string will be popped)
      for (; portion.size() != 0; portion.pop_back()) {
        std::vector<std::string> doc;
        boost::split(doc, portion.back(), boost::is_any_of(" \t\r"));
        if (doc.size() <= 1)
          continue;
        const int default_class = 0;
        const int unusual_class = 1;
        int current_class = default_class;
        for (int j = 1; j < static_cast<int>(doc.size() - 1); ++j) {
          if (doc[j][0] == '|') {
            if (strcmp(doc[j].c_str(), "|@default_class") == 0)
              current_class = default_class;
            else
              current_class = unusual_class;
            continue;
          }
          if (current_class != default_class)
            continue;

          auto first_token = vocab_dictionary_.find(doc[j]);
          if (first_token == vocab_dictionary_.end())
            continue;

          {
            int current_class = default_class;
            int first_token_id = first_token->second;
            int not_a_word_counter = 0;
            // if there are some words beginnig on '|' in a text the window
            // should be extended
            for (int k = 1; k <= window_width_ + not_a_word_counter && j + k < static_cast<int>(doc.size()); ++k) {
              // ToDo: write macro here
              if (doc[j + k][0] == '|') {
                if (strcmp(doc[j + k].c_str(), "|@default_class") == 0)
                  current_class = default_class;
                else
                  current_class = unusual_class;
                ++not_a_word_counter;
                continue;
              }
              if (current_class != default_class)
                continue;

              auto second_token = vocab_dictionary_.find(doc[j + k]);
              if (second_token == vocab_dictionary_.end())
                continue;
              int second_token_id = second_token->second;

              SavePairOfTokens(first_token_id, second_token_id, portion.size(), cooc_map);
              SavePairOfTokens(second_token_id, first_token_id, portion.size(), cooc_map);
              total_num_of_pairs_ += 2;
            }
          }
        }
      }

      if (!cooc_map.empty())
        UploadCooccurrenceBatchOnDisk(cooc_map);
    }
  };

  std::vector<std::shared_future<void>> tasks;
  for (int i = 0; i < num_of_threads_; ++i)
    tasks.push_back(std::move(std::async(std::launch::async, func)));
  for (int i = 0; i < num_of_threads_; ++i)
    tasks[i].get();
}

int CooccurrenceDictionary::CooccurrenceBatchQuantity() {
  return vector_of_batches_.size();
}

void CooccurrenceDictionary::ReadAndMergeCooccurrenceBatches() {
  auto CompareBatches = [](const std::unique_ptr<CooccurrenceBatch>& left,
                           const std::unique_ptr<CooccurrenceBatch>& right) {
    return left->current_cell_.first_token_id > right->current_cell_.first_token_id;
  };
  for (int i = 0; i < static_cast<int>(vector_of_batches_.size()) && i < max_num_of_open_files_ - 3; ++i) {
    OpenBatchInputFile(*(vector_of_batches_[i]));
    vector_of_batches_[i]->ReadCell();
  }
  for (int i = max_num_of_open_files_ - 3; i < static_cast<int>(vector_of_batches_.size()); ++i) {
    OpenBatchInputFile(*(vector_of_batches_[i]));
    vector_of_batches_[i]->ReadCell();
    CloseBatchInputFile(*(vector_of_batches_[i]));
  }
  std::make_heap(vector_of_batches_.begin(), vector_of_batches_.end(), CompareBatches);

  // This buffer won't hold more than 1 element, because another element
  // means another first_token_id => all the data linked with current
  // first_token_id can be filtered and be ready to be written in
  // resulting file.
  ResultingBuffer res(cooc_min_tf_, cooc_min_df_, calculate_tf_cooc_,
          calculate_df_cooc_, calculate_tf_ppmi_, calculate_df_ppmi_,
          calculate_ppmi_, total_num_of_pairs_, total_num_of_documents_,
          cooc_tf_file_path_, cooc_df_file_path_, ppmi_tf_file_path_,
          ppmi_df_file_path_);
  open_files_counter_ += 2;

  // Standard k-way merge as external sort
  while (!vector_of_batches_.empty()) {
    // It's guaranteed that batches aren't empty (look ParseVowpalWabbit func)
    res.AddInBuffer(*vector_of_batches_[0]);
    std::pop_heap(vector_of_batches_.begin(), vector_of_batches_.end(), CompareBatches);
    if (!vector_of_batches_.back()->in_batch_.is_open())
      OpenBatchInputFile(*(vector_of_batches_.back()));
    // if there are some data to read ReadCell reads it and returns true, else
    // return false
    if (vector_of_batches_.back()->ReadCell()) {
      if (max_num_of_open_files_ == open_files_counter_)
        CloseBatchInputFile(*(vector_of_batches_.back()));
      std::push_heap(vector_of_batches_.begin(), vector_of_batches_.end(), CompareBatches);
    } else {
      if (IsOpenBatchInputFile(*(vector_of_batches_.back())))
        CloseBatchInputFile(*(vector_of_batches_.back()));
      vector_of_batches_.pop_back();
    }
  }
  res.CheckPreviousCell(res.list_of_tf_records_, cooc_min_tf_);
  res.CheckPreviousCell(res.list_of_df_records_, cooc_min_df_);
  if (calculate_ppmi_) {
    res.BuildFreqDictionary();
    if (calculate_tf_ppmi_) {
      res.CalculateTfPpmi();
      res.WritePpmiInResultingFile(res.list_of_tf_records_, res.ppmi_tf_dict_);
    }
    if (calculate_df_ppmi_) {
      res.CalculateDfPpmi();
      res.WritePpmiInResultingFile(res.list_of_df_records_, res.ppmi_df_dict_);
    }
  }
  if (write_tf_cooc_)
    res.WriteCoocInResultingFile(res.list_of_tf_records_, res.cooc_tf_dict_);
  if (write_df_cooc_)
    res.WriteCoocInResultingFile(res.list_of_df_records_, res.cooc_df_dict_);
}

// ToDo: finish for mac, win, unices not caught below
// Note: user has to have some special headers on system (look in includes of)
// cooccurrence_dictionary.h
int CooccurrenceDictionary::SetItemsPerBatch() {
  // Here is a tool that allows to define an otimal value of documents that
  // should be load in ram. It depends on size of ram, window width and
  // num of threads (because every thread holds its batch of documents)
  const int default_value = 5000;
  const double percent_of_ram = 0.4;
  const long long std_ram_size = 4025409536; // 4 Gb
  const int std_window_width = 10;
  const int std_num_of_threads = 2;
  long long totalram;
#if defined(_WIN32)
  return default_value * percent_of_ram;
#elif defined(__APPLE__)
  #if defined(TARGET_OS_MAC)
    return default_value * percent_of_ram;
  #endif
#elif defined(__linux__) || defined(__linux) || defined(linux)
  struct sysinfo ex;
  sysinfo(&ex);
  totalram = ex.totalram;
#elif defined(__unix__) // all unices not caught above
  return default_value * percent_of_ram;
#else // other platforms
  return default_value * percent_of_ram;
#endif
  return static_cast<double>(std_window_width) / window_width_ *
      totalram / std_ram_size * std_num_of_threads / num_of_threads_ *
      default_value * percent_of_ram;
}

void CooccurrenceDictionary::UploadCooccurrenceBatchOnDisk(CoocMap& cooc_map) {
  std::unique_ptr<CooccurrenceBatch> batch(CreateNewCooccurrenceBatch());
  OpenBatchOutputFile(*batch);
  for (auto iter = cooc_map.begin(); iter != cooc_map.end(); ++iter) {
    batch->FormNewCell(iter);
    batch->WriteCell();
  }
  CloseBatchOutputFile(*batch);
  vector_of_batches_.push_back(std::move(batch));
}

CooccurrenceInfo CooccurrenceDictionary::FormInitialCoocInfo(int doc_id) {
  CooccurrenceInfo res;
  res.cooc_df = 1;
  res.cooc_tf = 1;
  res.prev_doc_id = doc_id;
  return res;
}

FirstTokenInfo CooccurrenceDictionary::FormInitialFirstTokenInfo(int doc_id) {
  FirstTokenInfo res;
  res.doc_num = 1;
  res.prev_doc_id = doc_id;
  return res;
}

void CooccurrenceDictionary::SavePairOfTokens(int first_token_id,
        int second_token_id, int doc_id, CoocMap& cooc_map) {
  auto map_record = cooc_map.find(first_token_id);
  if (map_record == cooc_map.end())
    AddInCoocMap(first_token_id, second_token_id, doc_id, cooc_map);
  else {
    ModifyCoocMapNode(second_token_id, doc_id, map_record->second);
  }
}

void CooccurrenceDictionary::AddInCoocMap(int first_token_id,
        int second_token_id, int doc_id, CoocMap& cooc_map) {
  CooccurrenceInfo new_cooc_info = FormInitialCoocInfo(doc_id);
  FirstTokenInfo new_first_token = FormInitialFirstTokenInfo(doc_id);
  SecondTokenInfo new_second_token;
  new_second_token.insert(std::make_pair(second_token_id, new_cooc_info));
  cooc_map.insert(std::make_pair(first_token_id,
              std::make_pair(new_first_token, new_second_token)));
}

void CooccurrenceDictionary::ModifyCoocMapNode(int second_token_id,
        int doc_id, std::pair<FirstTokenInfo, SecondTokenInfo>& map_node) {
  if (map_node.first.prev_doc_id != doc_id) {
    map_node.first.prev_doc_id = doc_id;
    ++map_node.first.doc_num;
  }
  SecondTokenInfo& second_token_map = map_node.second;
  auto iter = second_token_map.find(second_token_id);
  if (iter == second_token_map.end()) {
    CooccurrenceInfo new_cooc_info = FormInitialCoocInfo(doc_id);
    second_token_map.insert(std::pair<int, CooccurrenceInfo>(second_token_id, new_cooc_info));
  } else {
    ++iter->second.cooc_tf;
    if (iter->second.prev_doc_id != doc_id) {
      iter->second.prev_doc_id = doc_id;
      ++iter->second.cooc_df;
    }
  }
}

CooccurrenceBatch* CooccurrenceDictionary::CreateNewCooccurrenceBatch() {
  return new CooccurrenceBatch(path_to_batches_);
}

void CooccurrenceDictionary::OpenBatchInputFile(CooccurrenceBatch& batch) {
  if (open_files_counter_ >= max_num_of_open_files_)
    throw "Max number of open files achieved, can't open more";
  ++open_files_counter_;
  batch.in_batch_.open(batch.filename_, std::ios::in);
  batch.in_batch_.seekg(batch.in_batch_offset_);
}

void CooccurrenceDictionary::OpenBatchOutputFile(CooccurrenceBatch& batch) {
  if (open_files_counter_ >= max_num_of_open_files_)
    throw "Max number of open files achieved, can't open more";
  ++open_files_counter_;
  batch.out_batch_.open(batch.filename_, std::ios::out);
}

bool CooccurrenceDictionary::IsOpenBatchInputFile(CooccurrenceBatch& batch) {
  return batch.in_batch_.is_open();
}

void CooccurrenceDictionary::CloseBatchInputFile(CooccurrenceBatch& batch) {
  --open_files_counter_;
  batch.in_batch_offset_ = batch.in_batch_.tellg();
  batch.in_batch_.close();
}

void CooccurrenceDictionary::CloseBatchOutputFile(CooccurrenceBatch& batch) {
  --open_files_counter_;
  batch.out_batch_.close();
}

// ******************Methods of class CooccurrenceBatch*******************

CooccurrenceBatch::CooccurrenceBatch(const std::string& path_to_batches) {
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  fs::path batch(boost::lexical_cast<std::string>(uuid));
  fs::path full_filename = fs::path(path_to_batches) / batch;
  filename_ = full_filename.string();
  in_batch_offset_ = 0;
}

void CooccurrenceBatch::FormNewCell(const CoocMap::iterator& map_node) {
  // Every cooccurrence batch is divided into cells as folowing:
  // Different cells have different first token id values.
  // One cell contain records with euqal first token id
  // One cooccurrence batch can't hold 2 or more cells in ram simultaneously
  // Other cells are stored in output file
  current_cell_.first_token_id = map_node->first;
  current_cell_.num_of_documents = map_node->second.first.doc_num;
  SecondTokenInfo& second_token_map = map_node->second.second;
  current_cell_.num_of_records = static_cast<int>(second_token_map.size());
  current_cell_.batch_records.resize(current_cell_.num_of_records);
  int i = 0;
  for (auto iter = second_token_map.begin(); iter != second_token_map.end(); ++iter, ++i) {
    current_cell_.batch_records[i].first = iter->first;
    current_cell_.batch_records[i].second.cooc_tf = iter->second.cooc_tf;
    current_cell_.batch_records[i].second.cooc_df = iter->second.cooc_df;
  }
}

// ToDo: write normal serialization
void CooccurrenceBatch::WriteCell() {
  // Cells are written in following form: first line consists of
  // first token id, number of documents in which the folowing first token
  // occurred, num of records
  // The second line consists of records, which are separeted with a
  // space and numbers in these records are separeted the same
  std::stringstream ss;
  ss << current_cell_.first_token_id << ' ';
  ss << current_cell_.num_of_documents << ' ';
  ss << current_cell_.num_of_records << std::endl;
  for (int i = 0; i < static_cast<int>(current_cell_.batch_records.size()); ++i) {
    ss << current_cell_.batch_records[i].first << ' ';
    ss << current_cell_.batch_records[i].second.cooc_tf << ' ';
    ss << current_cell_.batch_records[i].second.cooc_df << ' ';
  }
  ss << std::endl;
  out_batch_ << ss.str();
}

bool CooccurrenceBatch::ReadCellHeader() {
  // If there is a cell header in the batch then there are records and Batch
  // isn't empty yet.
  // Result of the function is true if the batch isn't empty
  std::string str;
  getline(in_batch_, str);
  std::stringstream ss(str);
  ss >> current_cell_.first_token_id;
  ss >> current_cell_.num_of_documents;
  ss >> current_cell_.num_of_records;
  if (!in_batch_.eof())
    return true;
  else
    return false;
}

void CooccurrenceBatch::ReadRecords() {
  // It's not good if there are no records in batch after header
  if (in_batch_.eof())
    throw "Error while reading from batch. File is corrupted";
  std::string str;
  getline(in_batch_, str);
  std::stringstream ss(str);
  current_cell_.tf_records.clear();
  current_cell_.df_records.clear();
  for (int i = 0; i < static_cast<int>(current_cell_.num_of_records); ++i) {
    OutputInfo tmp_tf;
    OutputInfo tmp_df;
    int second_token_id;
    ss >> second_token_id;
    ss >> tmp_tf.cooc_value;
    ss >> tmp_df.cooc_value;
    tmp_tf.ppmi = 0.0;
    tmp_df.ppmi = 0.0;
    current_cell_.tf_records.push_back(std::make_pair(second_token_id, tmp_tf));
    current_cell_.df_records.push_back(std::make_pair(second_token_id, tmp_df));
  }
}

bool CooccurrenceBatch::ReadCell() {
  // As in ReadCellHeader function result here is true if batch isn't empty
  // If it's empty reading from this batch stops
  if (ReadCellHeader()) {
    ReadRecords();
    return true;
  }
  return false;
}

// ******************Methods of class ResultingBuffer*******************

ResultingBuffer::ResultingBuffer(const int cooc_min_tf, const int cooc_min_df,
    const bool calculate_tf_cooc, const bool calculate_df_cooc,
    const bool calculate_tf_ppmi, const bool calculate_df_ppmi,
    const bool calculate_ppmi, const long long total_num_of_pairs,
    const int total_num_of_documents,
    const std::string& cooc_tf_file_path,
    const std::string& cooc_df_file_path,
    const std::string& ppmi_tf_file_path,
    const std::string& ppmi_df_file_path) : cooc_min_tf_(cooc_min_tf),
        cooc_min_df_(cooc_min_df), calculate_tf_cooc_(calculate_tf_cooc),
        calculate_df_cooc_(calculate_df_cooc),
        calculate_tf_ppmi_(calculate_tf_ppmi),
        calculate_df_ppmi_(calculate_df_ppmi),
        calculate_ppmi_(calculate_ppmi),
        total_num_of_pairs_(total_num_of_pairs),
        total_num_of_documents_(total_num_of_documents),
        filebuf_size_(8500) {
  if (calculate_tf_cooc_) {
    cooc_tf_dict_.open(cooc_tf_file_path, std::ios::out);
    if (!cooc_tf_dict_.good())
      throw "Failed to create a file in the working directory";
  }
  if (calculate_df_cooc_) {
    cooc_df_dict_.open(cooc_df_file_path, std::ios::out);
    if (!cooc_df_dict_.good())
      throw "Failed to create a file in the working directory";
  }
  if (calculate_tf_ppmi_) {
    ppmi_tf_dict_.open(ppmi_tf_file_path, std::ios::out);
    if (!ppmi_tf_dict_.good())
      throw "Failed to create a file in the working directory";
  }
  if (calculate_df_ppmi_) {
    ppmi_df_dict_.open(ppmi_df_file_path, std::ios::out);
    if (!ppmi_df_dict_.good())
      throw "Failed to create a file in the working directory";
  }
}

void ResultingBuffer::AddInBuffer(const CooccurrenceBatch& batch) {
  if (list_of_tf_records_.size() != 0 &&
      list_of_tf_records_.back().first == batch.current_cell_.first_token_id)
    MergeWithExistingCell(list_of_tf_records_,
            batch.current_cell_.tf_records,
            batch.current_cell_.num_of_documents);
  else {
    CheckPreviousCell( list_of_tf_records_, cooc_min_tf_);
    AddNewCellInBuffer(list_of_tf_records_,
        batch.current_cell_.tf_records, batch.current_cell_.first_token_id,
        batch.current_cell_.num_of_documents);
  }
  if (list_of_df_records_.size() != 0 &&
      list_of_df_records_.back().first == batch.current_cell_.first_token_id)
    MergeWithExistingCell(list_of_df_records_,
            batch.current_cell_.df_records,
            batch.current_cell_.num_of_documents);
  else {
    CheckPreviousCell( list_of_df_records_, cooc_min_df_);
    AddNewCellInBuffer(list_of_df_records_,
        batch.current_cell_.df_records, batch.current_cell_.first_token_id,
        batch.current_cell_.num_of_documents);
  }
}

// ToDo: replace list with forward list
void ResultingBuffer::MergeWithExistingCell(OutputList& list_of_records,
        const OutputRecords& batch_records,
        const int num_of_documents_from_batch_cell) {
  OutputRecords new_list;
  OutputRecords& last_records = list_of_records.back().second.second;
  auto fi_iter = last_records.begin();
  auto se_iter = batch_records.begin();
  while (fi_iter != last_records.end() && se_iter != batch_records.end()) {
    if (fi_iter->first == se_iter->first) { // first is first_token_id
      OutputInfo tmp;
      tmp.cooc_value = fi_iter->second.cooc_value + se_iter->second.cooc_value;
      tmp.ppmi = 0.0;
      new_list.push_back(std::make_pair(fi_iter->first, tmp));
      ++fi_iter;
      ++se_iter;
    } else if (fi_iter->first < se_iter->first)
      new_list.push_back(*(fi_iter++));
    else
      new_list.push_back(*(se_iter++));
  }
  std::copy(fi_iter, last_records.end(),  std::back_inserter(new_list));
  std::copy(se_iter, batch_records.end(), std::back_inserter(new_list));
  int first_token_id = list_of_records.back().first;
  int new_num_of_documents = list_of_records.back().second.first +
              num_of_documents_from_batch_cell;
  list_of_records.pop_back();
  list_of_records.push_back(std::make_pair(first_token_id,
              std::make_pair(new_num_of_documents, new_list)));
}

void ResultingBuffer::CheckPreviousCell(OutputList& list_of_records,
        const int cooc_min_value) {
  if (list_of_records.empty())
    return;
  OutputRecords& last_records = list_of_records.back().second.second;
  for (auto iter = last_records.begin(); iter != last_records.end(); ) {
    if (iter->second.cooc_value < cooc_min_value)
      last_records.erase(iter++);
    else
      ++iter;
  }
  if (last_records.empty())
    list_of_records.pop_back();
}

void ResultingBuffer::AddNewCellInBuffer(OutputList& list_of_records,
        const OutputRecords& batch_records, const int first_token_id,
        const int num_of_documents) {
        /* num of documents correspondes to first token*/
  list_of_records.push_back(std::make_pair(first_token_id,
              std::make_pair(num_of_documents, batch_records)));
}

// ToDo: think how to fill it simultaniously for tf and df
void ResultingBuffer::BuildFreqDictionary() {
  for (auto iter1 = list_of_tf_records_.begin(); iter1 != list_of_tf_records_.end(); ++iter1) {
    long long token_freq = 0;
    OutputRecords& internal_list = iter1->second.second;
    for (auto iter2 = internal_list.begin(); iter2 != internal_list.end(); ++iter2) {
      token_freq += iter2->second.cooc_value;
    }
    // This Dictionary provides access to information about token (e.g. total
    // number times this token occurred in text, total number of documents
    // where the folowing token occurred)
    freq_dictionary_.insert(std::make_pair(iter1->first,
           std::make_pair(token_freq, 0))).second; // 0 will be replaced later
  }
  for (auto iter1 = list_of_df_records_.begin(); iter1 != list_of_df_records_.end(); ++iter1) {
    auto node = freq_dictionary_.find(iter1->first);
    if (node == freq_dictionary_.end())
      freq_dictionary_.insert(std::make_pair(iter1->first,
              std::make_pair(0, iter1->second.first))).second;
    else
      node->second.second = iter1->second.first;
  }
}

void ResultingBuffer::CalculateTfPpmi() {
  for (auto iter1 = list_of_tf_records_.begin(); iter1 != list_of_tf_records_.end(); ++iter1) {
    OutputRecords& internal_list = iter1->second.second;
    for (auto iter2 = internal_list.begin(); iter2 != internal_list.end(); ++iter2) {
      // ToDo: make some experiments of calculation
      // pmi(u, v) = (n / n_v) / (n_u / n_uv)
      /*{
        std::cout << '(' << iter1->first << ',' << iter2->first << ")\n";
        std::cout << "n = " << total_num_of_pairs_ << std::endl;
        std::cout << "n_v = " << freq_dictionary_.find(iter1->first)->second.first << std::endl;
        std::cout << "n_u = " << freq_dictionary_.find(iter2->first)->second.first << std::endl;
        std::cout << "n_uv = " << iter2->second.cooc_value << std::endl << std::endl;
      }*/
      double pmi = (static_cast<double>(total_num_of_pairs_) /
          freq_dictionary_.find(iter2->first)->second.first) /
          (freq_dictionary_.find(iter1->first)->second.first /
          static_cast<double>(iter2->second.cooc_value));
      // initially all the ppmi values are 0.0
      // (look CooccurrenceBatch::ReadRecords)
      if (pmi > 1.0) {
        iter2->second.ppmi = log(pmi);
      }
    }
  }
}

void ResultingBuffer::CalculateDfPpmi() {
  for (auto iter1 = list_of_df_records_.begin(); iter1 != list_of_df_records_.end(); ++iter1) {
    OutputRecords& internal_list = iter1->second.second;
    for (auto iter2 = internal_list.begin(); iter2 != internal_list.end(); ++iter2) {
      // ToDo: make some experiments of calculation
      // pmi(u, v) = (n / n_v) / (n_u / n_uv)
      /*{
        std::cout << '(' << iter1->first << ',' << iter2->first << ")\n";
        std::cout << "n = " << total_num_of_documents_ << std::endl;
        std::cout << "n_u = " << freq_dictionary_.find(iter1->first)->second.second << std::endl;
        std::cout << "n_v = " << freq_dictionary_.find(iter2->first)->second.second << std::endl;
        std::cout << "n_uv = " << iter2->second.cooc_value << std::endl << std::endl;
      }*/
      double pmi = (static_cast<double>(total_num_of_documents_) /
          freq_dictionary_.find(iter2->first)->second.second) /
          (freq_dictionary_.find(iter1->first)->second.second /
          static_cast<double>(iter2->second.cooc_value));
      // initially all the ppmi values are 0.0
      // (look CooccurrenceBatch::ReadRecords)
      if (pmi > 1.0) {
        iter2->second.ppmi = log(pmi);
      }
    }
  }
}

void ResultingBuffer::WriteCoocInResultingFile(OutputList& list_of_records,
        std::ofstream& cooc_dict) {
  std::stringstream filebuf;
  for (auto iter1 = list_of_records.begin();
           iter1 != list_of_records.end(); ++iter1) {
    OutputRecords& internal_list = iter1->second.second;
    for (auto iter2 = internal_list.begin();
             iter2 != internal_list.end(); ++iter2) {
      if (iter1->first != iter2->first)
        filebuf << iter1->first << " " << iter2->first << " "
                << iter2->second.cooc_value << std::endl;
      if (filebuf.tellg() > filebuf_size_)
        cooc_dict << filebuf.str();
    }
  }
  cooc_dict << filebuf.str();
}

void ResultingBuffer::WritePpmiInResultingFile(OutputList& list_of_records,
        std::ofstream& ppmi_dict) {
  std::stringstream filebuf;
  for (auto iter1 = list_of_records.begin();
           iter1 != list_of_records.end(); ++iter1) {
    OutputRecords& internal_list = iter1->second.second;
    for (auto iter2 = internal_list.begin();
             iter2 != internal_list.end(); ++iter2) {
      if (iter1->first < iter2->first)
        filebuf << iter1->first << " " << iter2->first << " "
                << iter2->second.ppmi << std::endl;
      if (filebuf.tellg() > filebuf_size_)
        ppmi_dict << filebuf.str();
    }
  }
  ppmi_dict << filebuf.str();
}
