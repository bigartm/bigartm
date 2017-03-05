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

void CooccurrenceBatch::FormNewCell(std::map<int, CoocMap>::iterator& map_node) {
  cell_.first_token_id = map_node->first;
  cell_.records.resize(0);
  for (auto iter = (map_node->second).begin(); iter != (map_node->second).end(); ++iter) {
    Triple tmp;
    tmp.cooc_value = iter->second.cooc_value;
    tmp.doc_quan   = iter->second.doc_quan;
    tmp.second_token_id = iter->first;
    cell_.records.push_back(tmp);
  }
  cell_.num_of_triples = cell_.records.size();
}

void CooccurrenceBatch::ReadRecords() {
  // It's not good if there are no records in batch after header
  if (in_batch_.eof()) {
    throw "Error while reading from batch. File is corrupted";
  }
  cell_.records.resize(cell_.num_of_triples);
  for (int i = 0; i < static_cast<int>(cell_.records.size()); ++i) {
    in_batch_ >> cell_.records[i].cooc_value;
    in_batch_ >> cell_.records[i].doc_quan;
    in_batch_ >> cell_.records[i].second_token_id;
  }
  //in_batch_.read(reinterpret_cast<char*>(&cell_.records[0]), sizeof(Triple) * cell_.num_of_triples);
}

void CooccurrenceBatch::WriteCell() {
  out_batch_ << cell_.first_token_id << ' ';
  //out_batch_.write(reinterpret_cast<char*>(&cell_.first_token_id), sizeof cell_.first_token_id);
  out_batch_ << cell_.num_of_triples << std::endl;
  //out_batch_.write(reinterpret_cast<char*>(&cell_.num_of_triples), sizeof cell_.num_of_triples);
  for (int i = 0; i < static_cast<int>(cell_.records.size()); ++i) {
    out_batch_ << cell_.records[i].cooc_value << ' ';
    out_batch_ << cell_.records[i].doc_quan << ' ';
    out_batch_ << cell_.records[i].second_token_id << std::endl;
  }
  //out_batch_.write(reinterpret_cast<char*>(&cell_.records[0]), sizeof(Triple) * cell_.num_of_triples);
}

bool CooccurrenceBatch::ReadCell() {
  if (ReadCellHeader()) {
    ReadRecords();
    return true;
  }
  return false;
}

bool CooccurrenceBatch::ReadCellHeader() {
  in_batch_ >> cell_.first_token_id;
  //in_batch_.read(reinterpret_cast<char*>(&cell_.first_token_id), sizeof cell_.first_token_id);
  in_batch_ >> cell_.num_of_triples;
  //in_batch_.read(reinterpret_cast<char*>(&cell_.num_of_triples), sizeof cell_.num_of_triples);
  if (!in_batch_.eof())
    return true;
  else {
    cell_.first_token_id = -1;
    cell_.num_of_triples = 0;
    return false;
  }
}

CooccurrenceBatch::CooccurrenceBatch(const std::string& path_to_batches) {
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  fs::path batch(boost::lexical_cast<std::string>(uuid));
  fs::path full_filename = fs::path(path_to_batches) / batch;
  filename_ = full_filename.string();
  in_batch_offset_ = 0;
}

ResultingBuffer::ResultingBuffer(const int min_tf, const int min_df,
         const std::string& cooc_tf_file_path,
         const std::string& cooc_df_file_path, const bool& cooc_tf_flag,
         const bool& cooc_df_flag) {
  // No need to check if buffer's empty. At first usage new data will need
  // to be pushed while previous popped, but previous data doesn't exist
  // (see AddInBuffer and PopPreviousContent methods)
  cooc_min_tf_ = min_tf;
  cooc_min_df_ = min_df;
  calculate_cooc_tf_ = cooc_tf_flag;
  calculate_cooc_df_ = cooc_df_flag;
  first_token_id_ = -1;
  if (calculate_cooc_tf_) {
    cooc_tf_dict_.open(cooc_tf_file_path, std::ios::out);
    if (!cooc_tf_dict_.good())
      throw "Failed to create a file in the working directory";
  }
  if (calculate_cooc_df_) {
    cooc_df_dict_.open(cooc_df_file_path, std::ios::out);
    if (!cooc_df_dict_.good())
      throw "Failed to create a file in the working directory";
  }
}

ResultingBuffer::~ResultingBuffer() { PopPreviousContent(); }

void ResultingBuffer::AddInBuffer(const CooccurrenceBatch& batch) {
  if (first_token_id_ == batch.cell_.first_token_id) {
    MergeWithExistingCell(batch);
  } else {
    PopPreviousContent();
    AddNewCellInBuffer(batch);
  }
}

void ResultingBuffer::MergeWithExistingCell(const CooccurrenceBatch& batch) {
  std::vector<Triple> new_vector;
  auto fi_iter = rec_.begin();
  auto se_iter = batch.cell_.records.begin();
  while (fi_iter != rec_.end() && se_iter != batch.cell_.records.end()) {
    if (fi_iter->second_token_id == se_iter->second_token_id) {
      Triple tmp;
      tmp.second_token_id = fi_iter->second_token_id;
      tmp.cooc_value = fi_iter->cooc_value + se_iter->cooc_value;
      tmp.doc_quan = fi_iter->doc_quan + se_iter->doc_quan;
      new_vector.push_back(tmp);
      ++fi_iter, ++se_iter;
    } else if (fi_iter->second_token_id < se_iter->second_token_id)
      new_vector.push_back(*(fi_iter++));
    else
      new_vector.push_back(*(se_iter++));
  }
  std::copy(fi_iter, rec_.end(), std::back_inserter(new_vector));
  std::copy(se_iter, batch.cell_.records.end(), std::back_inserter(new_vector));
  rec_ = new_vector;
}

void ResultingBuffer::PopPreviousContent() {
  for (int i = 0; i < static_cast<int>(rec_.size()); ++i) {
    if (calculate_cooc_tf_ && rec_[i].cooc_value >= cooc_min_tf_)
      cooc_tf_dict_ << first_token_id_ << " " << rec_[i].second_token_id
               << " " << rec_[i].cooc_value << std::endl;
    if (calculate_cooc_df_ && rec_[i].doc_quan >= cooc_min_df_)
      cooc_df_dict_ << first_token_id_ << " " << rec_[i].second_token_id
               << " " << rec_[i].doc_quan << std::endl;
  }
}

void ResultingBuffer::AddNewCellInBuffer(const CooccurrenceBatch& batch) {
  first_token_id_ = batch.cell_.first_token_id;
  rec_ = batch.cell_.records;
}

CooccurrenceDictionary::CooccurrenceDictionary(const std::string& vw,
        const std::string& vocab, const std::string& cooc_tf_file,
        const std::string& cooc_df_file, const int wind_width,
        const int min_tf, const int min_df) {
  // This function works as follows:
  // 1. Get content from a vocab file and put it in dictionary
  // 2. Read Vowpal Wabbit file by portions, calculate co-occurrences for
  // every portion and save it (cooccurrence batch) on external storage
  // 3. Read from external storage all the cooccurrence batches piece by
  // piece and create resulting file with all co-occurrences

  window_width_ = wind_width;
  cooc_min_tf_ = min_tf;
  cooc_min_df_ = min_df;
  path_to_vocab_ = vocab;
  path_to_vw_ = vw;
  cooc_tf_file_path_ = cooc_tf_file;
  cooc_df_file_path_ = cooc_df_file;
  calculate_tf_cooc_ = cooc_tf_file_path_.size() ? true : false;
  calculate_df_cooc_ = cooc_df_file_path_.size() ? true : false;
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  fs::path dir(boost::lexical_cast<std::string>(uuid));
  if (fs::exists(dir))
    throw "Folder with uuid already exists";
  if (!fs::create_directory(dir))
    throw "Failed to create directory";
  path_to_batches_ = dir.string();
  open_files_counter_ = 0;
  // ToDo: set it depending on OS settings
  {
    max_num_of_open_files_ = 1000;
  }
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
      std::map<int, CoocMap> cooc_maps;

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
              if (first_token_id == second_token_id)
                continue;

              SavePairOfTokens(first_token_id, second_token_id, portion.size(), cooc_maps);
              SavePairOfTokens(second_token_id, first_token_id, portion.size(), cooc_maps);
            }
          }
        }
      }

      if (!cooc_maps.empty())
        UploadCooccurrenceBatchOnDisk(cooc_maps);
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
    return left->cell_.first_token_id > right->cell_.first_token_id;
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
  ResultingBuffer res(cooc_min_tf_, cooc_min_df_, cooc_tf_file_path_,
          cooc_df_file_path_, calculate_tf_cooc_, calculate_df_cooc_);
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
}

// ToDo: finish for unix, mac, win
int CooccurrenceDictionary::SetItemsPerBatch() {
  // Here is a tool that allows to define an otimal value of documents that
  // should be load in ram. It depends on size of ram, window width and
  // num of threads (because every thread holds its batch of documents)
  const int default_value = 9000;
  const double percent_of_ram = 0.5;
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
#elif defined(__linux__)
  struct sysinfo ex;
  sysinfo(&ex);
  totalram = ex.totalram;
#elif defined(__unix__) // all unices not caught above
  return default_value * percent_of_ram;
#endif
  return static_cast<double>(std_window_width) / window_width_ *
      totalram / std_ram_size * std_num_of_threads / num_of_threads_ *
      default_value * percent_of_ram;
}

void CooccurrenceDictionary::UploadCooccurrenceBatchOnDisk(std::map<int, CoocMap>& cooc) {
  std::unique_ptr<CooccurrenceBatch> batch(CreateNewCooccurrenceBatch());
  OpenBatchOutputFile(*batch);
  for (auto iter = cooc.begin(); iter != cooc.end(); ++iter) {
    batch->FormNewCell(iter);
    batch->WriteCell();
  }
  CloseBatchOutputFile(*batch);
  vector_of_batches_.push_back(std::move(batch));
}

CooccurrenceInfo CooccurrenceDictionary::FormInitialCoocInfo(int doc_id) {
  CooccurrenceInfo res;
  res.doc_quan = 1;
  res.cooc_value = 1;
  res.prev_doc_id = doc_id;
  return res;
}

void CooccurrenceDictionary::AddInCoocMap(int first_token_id,
        int second_token_id, int doc_id, std::map<int, CoocMap>& cooc_maps) {
  CooccurrenceInfo tmp = FormInitialCoocInfo(doc_id);
  std::map<int, CooccurrenceInfo> tmp_map;
  tmp_map.insert(  std::pair<int, CooccurrenceInfo>(second_token_id, tmp));
  cooc_maps.insert(std::pair<int, std::map<int, CooccurrenceInfo>>(first_token_id, tmp_map));
}

void CooccurrenceDictionary::ModifyCoocMapNode(int second_token_id,
        int doc_id, CoocMap& map_node) {
  auto iter = map_node.find(second_token_id);
  if (iter == map_node.end()) {
    CooccurrenceInfo tmp = FormInitialCoocInfo(doc_id);
    map_node.insert(std::pair<int, CooccurrenceInfo>(second_token_id, tmp));
  } else {
    ++iter->second.cooc_value;
    if (iter->second.prev_doc_id != doc_id) {
      iter->second.prev_doc_id = doc_id;
      ++iter->second.doc_quan;
    }
  }
}

void CooccurrenceDictionary::SavePairOfTokens(int first_token_id,
        int second_token_id, int doc_id, std::map<int, CoocMap>& cooc_maps) {
  auto map_record = cooc_maps.find(first_token_id);
  if (map_record == cooc_maps.end())
    AddInCoocMap(first_token_id, second_token_id, doc_id, cooc_maps);
  else
    ModifyCoocMapNode(second_token_id, doc_id, map_record->second);
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
