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

CooccurrenceBatch::CooccurrenceBatch(int batch_num, const char filemode, const std::string& disk_path) {
  std::stringstream ss;
  ss << std::setprecision(4) << batch_num << ".bin";
  std::string filename = ss.str();
  fs::path full_filename = fs::path(disk_path) / fs::path(filename);
  if (filemode == 'w') {
    out_batch_.open(full_filename.string());
    if (out_batch_.bad()) {
      std::cerr << "CooccurrenceBatch::CooccurrenceBatch: Failed to create batch\n";
      throw 1;
    }
  } else {
    in_batch_.open(full_filename.string());
    if (in_batch_.bad()) {
      std::cerr << "CooccurrenceBatch::CooccurrenceBatch: Failed to open existing batch\n";
      throw 1;
    }
  }
}

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

bool CooccurrenceBatch::ReadCellHeader() {
  in_batch_.read(reinterpret_cast<char*>(&cell_.first_token_id), sizeof cell_.first_token_id);
  in_batch_.read(reinterpret_cast<char*>(&cell_.num_of_triples), sizeof cell_.num_of_triples);
  if (!in_batch_.eof())
    return true;
  else {
    cell_.first_token_id = -1;
    cell_.num_of_triples = 0;
    return false;
  }
}

void CooccurrenceBatch::ReadRecords() {
  // It's not good if there are no records in batch after header
    if (in_batch_.eof()) {
    std::cerr << "Error while reading from batch. File is corrupted\n";
    throw 1;
  }
  cell_.records.resize(cell_.num_of_triples);
  in_batch_.read(reinterpret_cast<char*>(&cell_.records[0]), sizeof(Triple) * cell_.num_of_triples);
}

bool CooccurrenceBatch::ReadCell() {
  if (ReadCellHeader()) {
    ReadRecords();
    return true;
  }
  return false;
}

void CooccurrenceBatch::WriteCell() {
  out_batch_.write(reinterpret_cast<char*>(&cell_.first_token_id), sizeof cell_.first_token_id);
  out_batch_.write(reinterpret_cast<char*>(&cell_.num_of_triples), sizeof cell_.num_of_triples);
  out_batch_.write(reinterpret_cast<char*>(&cell_.records[0]), sizeof(Triple) * cell_.num_of_triples);
}

BatchManager::BatchManager() {
  batch_quan_ = 0;
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  fs::path dir(boost::lexical_cast<std::string>(uuid));
  if (fs::exists(dir)) {
    std::cerr << "Folder with name Co-occurrenceBatch already exists in the working directory, please replace it\n";
    throw 1;
  }
  if (!fs::create_directory(dir)) {
    std::cerr << "Failed to create directory\n";
    throw 1;
  }
  path_to_batches_ = dir.string();
}

BatchManager::~BatchManager() { fs::remove_all(path_to_batches_); }

int BatchManager::GetBatchQuan() { return batch_quan_; }

CooccurrenceBatch* BatchManager::CreateNewBatch() {
  if (batch_quan_ < max_batch_quan_)
    return new CooccurrenceBatch(batch_quan_++, 'w', path_to_batches_);
  else {
    std::cerr << "Too many batches, maximal number of batches = "
              << max_batch_quan_ << std::endl;
    throw 1;
  }
}

CooccurrenceBatch* BatchManager::OpenExistingBatch(int batch_num) {
  return new CooccurrenceBatch(batch_num, 'r', path_to_batches_);
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

// Note: here is cast to int and comparison of floats
void ResultingBuffer::PopPreviousContent() {
  for (int i = 0; i < static_cast<int>(rec_.size()); ++i) {
    if (calculate_cooc_tf_ && rec_[i].cooc_value >= cooc_min_tf_)
      cooc_tf_dict_ << first_token_id_ << " " << rec_[i].second_token_id
               << " " << static_cast<int>(rec_[i].cooc_value) << std::endl;
    if (calculate_cooc_df_ && rec_[i].doc_quan >= cooc_min_df_)
      cooc_df_dict_ << first_token_id_ << " " << rec_[i].second_token_id
               << " " << rec_[i].doc_quan << std::endl;
  }
}

void ResultingBuffer::AddNewCellInBuffer(const CooccurrenceBatch& batch) {
  first_token_id_ = batch.cell_.first_token_id;
  rec_ = batch.cell_.records;
}

ResultingBuffer::ResultingBuffer(const float min_tf, const int min_df,
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
    if (cooc_tf_dict_.bad()) {
      std::cerr << "Failed to create a file in the working directory\n";
      throw 1;
    }
  }
  if (calculate_cooc_df_) {
    cooc_df_dict_.open(cooc_df_file_path, std::ios::out);
    if (cooc_df_dict_.bad()) {
      std::cerr << "Failed to create a file in the working directory\n";
      throw 1;
    }
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

CooccurrenceDictionary::CooccurrenceDictionary(const std::string& vw,
        const std::string& vocab, const std::string& cooc_tf_file,
        const std::string& cooc_df_file, const int wind_width,
        const float min_tf, const int min_df) {
  // This function works as follows:
  // 1. Get content from a vocab file and put it in dictionary
  // 2. Read Vowpal Wabbit file by portions, calculate co-occurrences for
  // every portion and save it (batch) on external storage
  // 3. Read from external storage all the batches piece by piece and create
  // resulting file with all co-occurrences

  try {
    path_to_vw_ = vw;
    if (path_to_vw_.size() == 0) {
      std::cerr << "input file in VowpalWabbit format not specified\n";
      throw 1;
    }
    path_to_vocab_ = vocab;
    if (path_to_vocab_.size() == 0) {
      std::cerr << "input file in UCI vocab format not specified\n";
      throw 1;
    }
    window_width_ = wind_width;
    cooc_min_tf_ = min_tf;
    cooc_min_df_ = min_df;
    cooc_tf_file_path_ = cooc_tf_file;
    cooc_df_file_path_ = cooc_df_file;
    calculate_tf_cooc_ = cooc_tf_file_path_.size() ? true : false;
    calculate_df_cooc_ = cooc_df_file_path_.size() ? true : false;
    // ToDo: set it depeding from RAM
    items_per_batch = 7000;
    max_num_of_batches = 4000;
    FetchVocab();
    if (vocab_dictionary_.size() > 1) {
      ReadVowpalWabbit();
      if (batch_manager_.GetBatchQuan() != 0)
        ReadAndMergeBatches();
    }
  } catch (...) {}
}

void CooccurrenceDictionary::FetchVocab() {
  // This func reads words from vocab, sets them unique id and collects pair
  // in dictionary
  std::ifstream vocab(path_to_vocab_, std::ios::in);
  if (!vocab.is_open()) {
    std::cerr << "Failed to open vocab\n";
    throw 1;
  }
  int last_token_id = 1;
  std::string str;

  while (true) {
    getline(vocab, str);
    if (vocab.eof())
      break;
    boost::algorithm::trim(str);
    if (!str.empty()) {
      bool inserted = vocab_dictionary_.insert(std::make_pair(str, last_token_id)).second;
      if (inserted)
        ++last_token_id;
    }
  }
}

void CooccurrenceDictionary::UploadBatchOnDisk(BatchManager& batch_manager,
        std::map<int, CoocMap>& cooc) {
  std::unique_ptr<CooccurrenceBatch> batch(batch_manager.CreateNewBatch());
  for (auto iter = cooc.begin(); iter != cooc.end(); ++iter) {
    batch->FormNewCell(iter);
    batch->WriteCell();
  }
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
  tmp_map.insert( std::pair<int, CooccurrenceInfo>(second_token_id, tmp));
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
  if (!vowpal_wabbit_doc.is_open()) {
    std::cerr << "Failed to open vocab\n";
    throw 1;
  }
  std::mutex read_lock, write_lock;

  auto func = [&]() {
    while (true) {
      std::vector<std::string> portion;

      {
        std::lock_guard<std::mutex> guard(read_lock);
        if (vowpal_wabbit_doc.eof())
          return;

        std::string str;
        while (static_cast<int>(portion.size()) < items_per_batch) {
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

        for (int j = 1; j < static_cast<int>(doc.size() - 1); ++j) {
          auto first_token = vocab_dictionary_.find(doc[j]);
          if (first_token == vocab_dictionary_.end())
            continue;
          int first_token_id = first_token->second;

          for (int k = 1; k <= window_width_ && j + k < static_cast<int>(doc.size()); ++k) {
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

      {
        std::lock_guard<std::mutex> guard(write_lock);
        if (!cooc_maps.empty())
          UploadBatchOnDisk(batch_manager_, cooc_maps);
      }
    }
  };

  int num_of_threads = std::thread::hardware_concurrency();
  if (num_of_threads == 0)
    num_of_threads = 1;
  std::vector<std::shared_future<void>> tasks;
  for (int i = 0; i < num_of_threads; ++i)
    tasks.push_back(std::move(std::async(std::launch::async, func)));
  for (int i = 0; i < num_of_threads; ++i)
    tasks[i].get();
}

void CooccurrenceDictionary::ReadAndMergeBatches() {
  auto CompareBatches = [](const std::unique_ptr<CooccurrenceBatch>& left,
                           const std::unique_ptr<CooccurrenceBatch>& right) {
    return left->cell_.first_token_id > right->cell_.first_token_id;
  };
  std::vector<std::unique_ptr<CooccurrenceBatch>> batch_queue;
  for (int i = 0; i < batch_manager_.GetBatchQuan(); ++i) {
    std::unique_ptr<CooccurrenceBatch> tmp(batch_manager_.OpenExistingBatch(i));
    batch_queue.push_back(std::move(tmp));
    batch_queue[i]->ReadCell();
  }
  std::make_heap(batch_queue.begin(), batch_queue.end(), CompareBatches);

  // This buffer won't hold more than 1 element, because another element
  // means another first_token_id => all the data linked with current
  // first_token_id can be filtered and be ready to be written in
  // resulting file.
  ResultingBuffer res(cooc_min_tf_, cooc_min_df_, cooc_tf_file_path_,
          cooc_df_file_path_, calculate_tf_cooc_, calculate_df_cooc_);

  // Standard k-way merge as external sort
  while (!batch_queue.empty()) {
    // It's guaranteed that batches aren't empty (look ParseVowpalWabbit func)
    res.AddInBuffer(*batch_queue[0]);
    std::pop_heap(batch_queue.begin(), batch_queue.end(), CompareBatches);
    if (batch_queue.back()->ReadCell())
      std::push_heap(batch_queue.begin(), batch_queue.end(), CompareBatches);
    else
      batch_queue.pop_back();
  }
}
