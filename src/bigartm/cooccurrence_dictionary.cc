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
#include <cassert>
#include <stdexcept>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/utility.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

namespace fs = boost::filesystem;

// ***************************** Methods of class CooccurrenceDictionary ********************

CooccurrenceDictionary::CooccurrenceDictionary(const int window_width,
    const int cooc_min_tf, const int cooc_min_df,
    const std::string& path_to_vocab, const std::string& path_to_vw,
    const std::string& cooc_tf_file_path,
    const std::string& cooc_df_file_path,
    const std::string& ppmi_tf_file_path,
    const std::string& ppmi_df_file_path, const int num_of_threads) :
        window_width_(window_width), cooc_min_tf_(cooc_min_tf),
        cooc_min_df_(cooc_min_df), path_to_vocab_(path_to_vocab),
        path_to_vw_(path_to_vw),
        cooc_tf_file_path_(cooc_tf_file_path),
        cooc_df_file_path_(cooc_df_file_path),
        ppmi_tf_file_path_(ppmi_tf_file_path),
        ppmi_df_file_path_(ppmi_df_file_path),
        open_files_counter_(0), max_num_of_open_files_(500), total_num_of_pairs_(0),
        total_num_of_documents_(0), documents_per_batch_(16000), num_of_threads_(num_of_threads) {
  // Calculation of token co-occurrence starts with this class

  calculate_tf_cooc_ = cooc_tf_file_path_.size() != 0;
  calculate_df_cooc_ = cooc_df_file_path_.size() != 0;
  calculate_tf_ppmi_ = ppmi_tf_file_path_.size() != 0;
  calculate_df_ppmi_ = ppmi_df_file_path_.size() != 0;
  calculate_ppmi_ = calculate_tf_ppmi_ || calculate_df_ppmi_;
  
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  fs::path dir(boost::lexical_cast<std::string>(uuid));
  if (fs::exists(dir)) {
    throw std::invalid_argument("Folder with uuid already exists");
  }
  if (!fs::create_directory(dir)) {
    throw std::invalid_argument("Failed to create directory");
  }

  path_to_batches_ = dir.string();
  if (num_of_threads_ == -1) {
    num_of_threads_ = std::thread::hardware_concurrency();
    if (num_of_threads_ == 0) {
      num_of_threads_ = 1;
    }
  }
  std::cout << "documents per batch = " << documents_per_batch_ << std::endl;
}

void CooccurrenceDictionary::FetchVocab() {
  // This function reads tokens from vocab, sets them unique id and collects pairs in dictionary
  // If there was modality label, takes only tokens from default class

  std::ifstream vocab(path_to_vocab_, std::ios::in);
  if (!vocab.good()) {
    throw std::invalid_argument("Failed to open vocab");
  }
  std::string str;
  for (int last_token_id = 0; getline(vocab, str); ++last_token_id) {
    boost::algorithm::trim(str);
    std::vector<std::string> strs;
    boost::split(strs, str, boost::is_any_of(" "));
    if (!strs[0].empty()) {
      if (strs.size() == 1 || strs[1] == "@default_class") { // check modality
        vocab_dictionary_.insert(std::make_pair(strs[0], last_token_id));
      }
    }
  }
  token_statistics_.resize(vocab_dictionary_.size()); // initialization of token_statistics_
}

int CooccurrenceDictionary::VocabDictionarySize() {
  return vocab_dictionary_.size();
}

// ToDo: decide how you are going to calculate variables for ppmi
// ToDo: decide how to calculate n_u if it takes all the pairs with token u (even with unnecesery another token)
void CooccurrenceDictionary::ReadVowpalWabbit() {
  // This function works as follows:
  // 1. Acquire lock for reading from vowpal wabbit file
  // 2. Read a portion (documents_per_batch) of documents from file and save it
  // in a local buffer (vector<string>)
  // 3. Release the lock
  // 4. Cut every document into tokens.
  // 5. Search for each pair of tokens in vocab dictionary and if they're valid (were found),
  // calculate their co-occurrences (absolute and documental) and othen statistics
  // for future pmi calculation (in how many pairs and documents a specific token occured,
  // total number of documents and pairs in collection).
  // Co-occurrence counters are stored in map of maps (on external level of indexing are first_token_id's
  // and on internal second_token_id's)
  // Statistics that refer to unique tokens is stored in vector called token_statistics_
  // 6. For each potion of documents create a batch (class CooccurrenceBatch) with co-occurrence statistics
  // Repeat 1-6 for all portions (can work in parallel for different portions)

  std::cout << "Step 1: creation of co-occurrence batches" << std::endl;
  std::string documents_processed = std::to_string(total_num_of_documents_);
  std::cout << "Documents processed: " << documents_processed << std::flush;
  std::ifstream vowpal_wabbit_doc(path_to_vw_, std::ios::in);
  if (!vowpal_wabbit_doc.is_open()) {
    throw std::invalid_argument("Failed to open vocab");
  }
  std::mutex read_lock;
  std::mutex stdout_lock;

  auto func = [&]() {
    while (true) { // Loop throgh portions.
      // Steps 1-3:
      std::vector<std::string> portion = ReadPortionOfDocuments(read_lock, vowpal_wabbit_doc);
      if (portion.empty()) {
        break;
      }
      total_num_of_documents_ += portion.size(); // statistics for ppmi

      // It will hold tf and df of pairs of tokens
      // Every pair of valid tokens (both exist in vocab) is saved in this storage
      // After walking through portion of documents all the statistics is dumped on disk
      // and then this storage is destroyed
      CooccurrenceStatisticsHolder cooc_stat_holder;

      // For every token from vocab keep the information about the last document this token occured in
      std::vector<unsigned> num_of_last_document_token_occured(vocab_dictionary_.size());

      // When the document is processed (element of portion vector),
      // memory for it can be freed by calling pop_back() from vector
      // (large string will be popped and destroyed)
      // portion.size() can be used as doc_id (it will in method SavePairOfTokens)
      for (; portion.size() != 0; portion.pop_back()) { // Loop through documents. Step 4:
        std::vector<std::string> doc;
        boost::split(doc, portion.back(), boost::is_any_of(" \t\r"));

        // Step 5:
        // 5.a) There are rules how to consider modalities: now only tokens of default_class are processed
        // Start loop from 1 because the zeroth element is document title
        int first_token_current_class = DEFAULT_CLASS;
        for (unsigned j = 1; j < doc.size(); ++j) { // Loop through tokens in document
          if (doc[j].empty()) {
            continue;
          }
          if (doc[j][0] == '|') {
            first_token_current_class = SetModalityLabel(doc[j]);
            continue;
          }
          if (first_token_current_class != DEFAULT_CLASS) {
            continue;
          }

          // 5.b) Check if a token is valid
          auto first_token = vocab_dictionary_.find(doc[j]);
          if (first_token == vocab_dictionary_.end()) {
            continue;
          }
          int first_token_id = first_token->second;

          // 5.c) Collect statistics for ppmi (in how many documents every token occured)
          // In the beginning the zeros, so for evey value of portion.size() these values aren't equal
          // and if it's the first document in the collection where a specific token occurred
          // values aren't equal and counter of douments for specific token is incremented
          if (num_of_last_document_token_occured[first_token_id] != portion.size()) {
            num_of_last_document_token_occured[first_token_id] = portion.size();
            ++token_statistics_[first_token_id].num_of_documents_token_occured_in;
          }

          // 5.d) Take windows_width tokens (parameter) to the right of the current one
          // If there are some words beginnig on '|' in a text the window should be extended
          // and it's extended using not_a_word_counter
          unsigned not_a_word_counter = 0;
          int second_token_current_class = DEFAULT_CLASS;
          for (unsigned k = 1; k <= window_width_ + not_a_word_counter && j + k < doc.size(); ++k) { // Loop through tokens in window
            if (doc[j + k].empty()) {
              continue;
            }
            if (doc[j + k][0] == '|') {
              second_token_current_class = SetModalityLabel(doc[j + k]);
              ++not_a_word_counter;
              continue;
            }
            if (second_token_current_class != DEFAULT_CLASS) {
              continue;
            }

            auto second_token = vocab_dictionary_.find(doc[j + k]);
            if (second_token == vocab_dictionary_.end()) {
              continue;
            }
            int second_token_id = second_token->second;

            // 5.e) When it's known these 2 tokens are valid, remember their co-occurrence
            // Here portion.size() is used to identify a document (it's unique id during one portion of documents)
            cooc_stat_holder.SavePairOfTokens(first_token_id, second_token_id, portion.size());
            cooc_stat_holder.SavePairOfTokens(second_token_id, first_token_id, portion.size());
            total_num_of_pairs_ += 2; // statistics for ppmi
          }
        }
      }

      if (!cooc_stat_holder.IsEmpty()) {
        // This function saves gathered statistics on disk
        // After saving on disk statistics from all the batches needs to be merged
        // This is implemented in ReadAndMergeCooccurrenceBatches(), so the next step is to call this method
        cooc_stat_holder.SortFirstTokens(); // Sorting is needed before storing all pairs of tokens on disk (it's for future agregation)
        cooc_stat_holder.SortSecondTokens();
        UploadOnDisk(cooc_stat_holder);
      }

      { // print number of documents which were precessed
        std::lock_guard<std::mutex> guard(stdout_lock);
        for (unsigned i = 0; i < documents_processed.size(); ++i) {
          std::cout << '\b';
        }
        documents_processed = std::to_string(total_num_of_documents_);
        std::cout << documents_processed << std::flush;
      }
    }
  };

  // Launch reading and storing pairs of tokens in parallel
  std::vector<std::shared_future<void>> tasks;
  for (int i = 0; i < num_of_threads_; ++i) {
    tasks.emplace_back(std::async(std::launch::async, func));
  }
  for (int i = 0; i < num_of_threads_; ++i) {
    tasks[i].get();
  }
  std::cout << '\n' << "Co-occurrence batches have been created" << std::endl;
}

std::vector<std::string> CooccurrenceDictionary::ReadPortionOfDocuments(std::mutex& read_lock,std::ifstream& vowpal_wabbit_doc) {
  std::vector<std::string> portion;
  std::lock_guard<std::mutex> guard(read_lock);
  if (vowpal_wabbit_doc.eof()) {
    return portion;
  }
  std::string str;
  while (portion.size() < documents_per_batch_) {
    getline(vowpal_wabbit_doc, str);
    if (vowpal_wabbit_doc.eof()) {
      break;
    }
    portion.push_back(str);
  }
  return portion;
}

int CooccurrenceDictionary::SetModalityLabel(std::string& modality_label) {
  if (modality_label == "|@default_class") {
    return DEFAULT_CLASS;
  } else {
    return UNUSUAL_CLASS;
  }
}

void CooccurrenceDictionary::UploadOnDisk(CooccurrenceStatisticsHolder& cooc_stat_holder) {
  // Uploading is implemented as folowing:
  // 1. Create a batch which is associated with a specific file on a disk
  // 2. For every first token id create an object Cell and for every second token
  // that co-occurred with first write it's id, cooc_tf, cooc_df
  // 3. Write the cell in output file and continue the cicle while there are 
  // first token ids in cooccurrence statistics holder 
  // Note that there can't be two cells stored in ram simultaniously
  // 4. Save batch in vector of objects

  std::unique_ptr<CooccurrenceBatch> batch(CreateNewCooccurrenceBatch());
  OpenBatchOutputFile(*batch);
  for (auto iter = cooc_stat_holder.storage_.begin(); iter != cooc_stat_holder.storage_.end(); ++iter) {
    batch->FormNewCell(iter);
    batch->WriteCell();
  }
  CloseBatchOutputFile(*batch);
  vector_of_batches_.push_back(std::move(batch));
}

CooccurrenceBatch* CooccurrenceDictionary::CreateNewCooccurrenceBatch() {
  return new CooccurrenceBatch(path_to_batches_);
}

void CooccurrenceDictionary::OpenBatchOutputFile(CooccurrenceBatch& batch) {
  assert(open_files_counter_ < max_num_of_open_files_);
  ++open_files_counter_;
  batch.out_batch_.open(batch.filename_, std::ios::out);
}

void CooccurrenceDictionary::CloseBatchOutputFile(CooccurrenceBatch& batch) {
  --open_files_counter_;
  batch.out_batch_.close();
}

int CooccurrenceDictionary::CooccurrenceBatchesQuantity() {
  return vector_of_batches_.size();
}

// ToDo: implement parallel merging (important!)
// ToDo: may be create a new function (k-way merge)
void CooccurrenceDictionary::ReadAndMergeCooccurrenceBatches() {
  std::cout << "Step 2: merging batches" << std::endl;

  // After that all the statistics has been gathered and saved in form of batches on disk, it
  // needs to be merged from batches into one storage (This is may be the most long-time part of co-occurrence gathering)
  // All batches has it's local buffer in operating memory (look the CooccurrenceBatch class realization)
  // Information in batches is stored in cells
  // Here's the k-way merge algorithm:
  // 1. Initially first cells of all the batches are read in their buffers
  // 2. Then batches are sorted (std::make_heap) by first_token_id of the cell
  // 3. Then a cell with the lowest first_token_id is extaracted and put in resulting buffer and the next cell
  // is read from corresponding batch
  // 4. If lowest first token id equals first token id of cell of this buffer, they are merged
  // Else the current cell is written in file and the new one is loaded
  // Writing and empting is done in order to keep low memory consumption
  // If there would be a need to calculate ppmi or other values which depend on co-occurrences
  // this data can be read back from the file

  // There are some comments in class constructor which explain how this buffer works
  ResultingBufferOfCooccurrences res(cooc_min_tf_, cooc_min_df_, calculate_tf_cooc_,
          calculate_df_cooc_, calculate_tf_ppmi_, calculate_df_ppmi_,
          calculate_ppmi_, total_num_of_pairs_, total_num_of_documents_,
          cooc_tf_file_path_, cooc_df_file_path_, ppmi_tf_file_path_, ppmi_df_file_path_, token_statistics_);
  open_files_counter_ += res.open_files_in_buf_;

  // Here's why CooccurrenceBatches are wrapped in unique_ptrs -- it's easier to move them (no need to reopen files)
  auto CompareBatches = [](const std::unique_ptr<CooccurrenceBatch>& left,
                           const std::unique_ptr<CooccurrenceBatch>& right) {
    return left->cell_.first_token_id > right->cell_.first_token_id;
  };
  auto iter = vector_of_batches_.begin();
  for (; iter != vector_of_batches_.end() && open_files_counter_ < max_num_of_open_files_ - 1; ++iter) {
    OpenBatchInputFile(**iter);
    (*iter)->ReadCell();
  }
  for (; iter != vector_of_batches_.end(); ++iter) {
    OpenBatchInputFile(**iter);
    (*iter)->ReadCell();
    CloseBatchInputFile(**iter);
  }
  std::make_heap(vector_of_batches_.begin(), vector_of_batches_.end(), CompareBatches);

  // k-way merge as external sort
  while (!vector_of_batches_.empty()) {
    // It's guaranteed that batches aren't empty (look ParseVowpalWabbit func)
    // Addition in buffer can cause writing exesting data in file and putting other data on it's place
    res.AddInBuffer(*vector_of_batches_[0]);
    std::pop_heap(vector_of_batches_.begin(), vector_of_batches_.end(), CompareBatches);
    if (!vector_of_batches_.back()->in_batch_.is_open()) {
      OpenBatchInputFile(*(vector_of_batches_.back()));
    }
    // if there are some data to read ReadCell reads it and returns true, else
    // returns false
    if (vector_of_batches_.back()->ReadCell()) {
      if (max_num_of_open_files_ == open_files_counter_) {
        CloseBatchInputFile(*(vector_of_batches_.back()));
      }
      std::push_heap(vector_of_batches_.begin(), vector_of_batches_.end(), CompareBatches);
    } else {
      if (IsOpenBatchInputFile(*(vector_of_batches_.back()))) {
        CloseBatchInputFile(*(vector_of_batches_.back()));
      }
      vector_of_batches_.pop_back();
    }
  }
  if (!res.cell_.records.empty()) {
    // unknown variables needed for ppmi calcualation are found here and during k-way merge
    res.WriteCoocFromBufferInFile();
  }
  // Files are explicitly closed here, because data needs to be pushed in files on this step
  if (calculate_tf_cooc_) {
    res.cooc_tf_dict_out_.close();
  }
  if (calculate_df_cooc_) {
    res.cooc_df_dict_out_.close();
  }

  std::cout << "Batches have been merged" << std::endl;
  if (calculate_ppmi_) {
    std::cout << "Step 3: start calculation ppmi" << std::endl;
    res.CalculateAndWritePpmi();
  }
  std::cout << "Ppmi's have been calculated" << std::endl;
}

void CooccurrenceDictionary::OpenBatchInputFile(CooccurrenceBatch& batch) {
  assert(open_files_counter_ < max_num_of_open_files_);
  ++open_files_counter_;
  batch.in_batch_.open(batch.filename_, std::ios::in);
  batch.in_batch_.seekg(batch.in_batch_offset_);
}

bool CooccurrenceDictionary::IsOpenBatchInputFile(CooccurrenceBatch& batch) {
  return batch.in_batch_.is_open();
}

void CooccurrenceDictionary::CloseBatchInputFile(CooccurrenceBatch& batch) {
  --open_files_counter_;
  batch.in_batch_offset_ = batch.in_batch_.tellg();
  batch.in_batch_.close();
}

CooccurrenceDictionary::~CooccurrenceDictionary() {
  fs::remove_all(path_to_batches_);
}

// ******************* Methods of class CooccurrenceStatisticsHolder ***********************

// This class stores temporarily added statistics about pairs of tokens (how often these pairs occurred in documents
// in a window, in how many documents they occurred together in a window)
// It's necessery to keep all them with a little mamory overhead
// That's why first tokens with some information about them and second tokens with info about co-occurrence
// are stored in vector
void CooccurrenceStatisticsHolder::SavePairOfTokens(int first_token_id, int second_token_id, unsigned doc_id) {
  // There are 2 levels of indexing
  // The first level keeps information about first token and the second level
  // about co-occurrence between the first and the second tokens
  // If first token id is known (exists in the structure), corresponding node should be modified
  // else it should be added to the structure
  int index1 = FindFirstToken(first_token_id);
  if (index1 == NOT_FOUND) {
    storage_.emplace_back(first_token_id);
    storage_.back().second_token_reference.emplace_back(second_token_id);
  } else {
    std::vector<SecondTokenAndCooccurrence>& second_tokens = storage_[index1].second_token_reference;
    int index2 = FindSecondToken(second_tokens, second_token_id);
    if (index2 == NOT_FOUND) {
      second_tokens.emplace_back(second_token_id);
    } else {
      SecondTokenAndCooccurrence& cooc_info = second_tokens[index2];
      if (cooc_info.last_doc_id != doc_id) {
        cooc_info.last_doc_id = doc_id;
        ++cooc_info.cooc_df;
      }
      ++cooc_info.cooc_tf;
    }
  }
}

int CooccurrenceStatisticsHolder::FindFirstToken(int first_token_id) {
  // Linear search
  for (unsigned i = 0; i < storage_.size(); ++i) {
    if (storage_[i].first_token_id == first_token_id) {
      return i;
    }
  }
  return NOT_FOUND;
}

int CooccurrenceStatisticsHolder::FindSecondToken(
      std::vector<CooccurrenceStatisticsHolder::SecondTokenAndCooccurrence>& second_tokens,
      int second_token_id) {
  // Linear search
  for (unsigned i = 0; i < second_tokens.size(); ++i) {
    if (second_tokens[i].second_token_id == second_token_id) {
      return i;
    }
  }
  return NOT_FOUND;
}

bool CooccurrenceStatisticsHolder::IsEmpty() {
  return storage_.empty();
}

void CooccurrenceStatisticsHolder::SortFirstTokens() {
  auto CompareFirstTokens = [](const CooccurrenceStatisticsHolder::FirstToken& left,
                               const CooccurrenceStatisticsHolder::FirstToken& right) {
    return left.first_token_id < right.first_token_id;
  };
  std::sort(storage_.begin(), storage_.end(), CompareFirstTokens);
}

void CooccurrenceStatisticsHolder::SortSecondTokens() {
  auto CompareSecondTokens = [](const CooccurrenceStatisticsHolder::SecondTokenAndCooccurrence& left,
                                const CooccurrenceStatisticsHolder::SecondTokenAndCooccurrence& right) {
    return left.second_token_id < right.second_token_id;
  };
  for (auto& iter : storage_) {
    std::sort(iter.second_token_reference.begin(), iter.second_token_reference.end(), CompareSecondTokens);
  }
}

// *********************** Methods of class CoccurrenceBatch ***************************

CooccurrenceBatch::CooccurrenceBatch(const std::string& path_to_batches) {
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  fs::path batch(boost::lexical_cast<std::string>(uuid));
  fs::path full_filename = fs::path(path_to_batches) / batch;
  filename_ = full_filename.string();
  in_batch_offset_ = 0;
}

void CooccurrenceBatch::FormNewCell(const std::vector<CooccurrenceStatisticsHolder::FirstToken>::iterator& cooc_stat_node) {
  // Here is initialization of a new cell
  // A cell consists on first_token_id, number of records it includes
  // Then records go, every reord consists on second_token_id, cooc_tf, cooc_df

  cell_.first_token_id = cooc_stat_node->first_token_id;
  // while reading from file it's necessery to know how many records to read
  std::vector<CooccurrenceStatisticsHolder::SecondTokenAndCooccurrence>& second_token_reference = cooc_stat_node->second_token_reference;
  cell_.num_of_records = second_token_reference.size();
  cell_.records.resize(cell_.num_of_records);
  int i = 0;
  for (auto iter = second_token_reference.begin(); iter != second_token_reference.end(); ++iter, ++i) {
    cell_.records[i].second_token_id = iter->second_token_id;
    cell_.records[i].cooc_tf = iter->cooc_tf;
    cell_.records[i].cooc_df = iter->cooc_df;
  }
}

void CooccurrenceBatch::WriteCell() {
  // Cells are written in following form: first line consists of first token id
  // and num of triples
  // the second line consists of numbers triples, which are separeted with a
  // space and numbers in these triples are separeted the same
  // stringstream is used for fast bufferized i/o operations
  std::stringstream ss;
  ss << cell_.first_token_id << ' ';
  ss << cell_.num_of_records << std::endl;
  for (unsigned i = 0; i < cell_.records.size(); ++i) {
    ss << cell_.records[i].second_token_id << ' ';
    ss << cell_.records[i].cooc_tf << ' ';
    ss << cell_.records[i].cooc_df << ' ';
  }
  ss << std::endl;
  out_batch_ << ss.str();
}

bool CooccurrenceBatch::ReadCell() {
  if (ReadCellHeader()) {
    ReadRecords();
    return true;
  }
  return false;
}

bool CooccurrenceBatch::ReadCellHeader() {
  // stringstream is used for fast bufferized i/o operations
  std::string str;
  getline(in_batch_, str);
  std::stringstream ss(str);
  ss >> cell_.first_token_id;
  ss >> cell_.num_of_records;
  if (!in_batch_.eof()) {
    return true;
  } else {
    return false;
  }
}

void CooccurrenceBatch::ReadRecords() {
  // It's not good if there are no records in batch after header
  if (in_batch_.eof()) {
    throw std::invalid_argument("Error while reading from batch. File is corrupted");
  }
  std::string str;
  getline(in_batch_, str);
  // stringstream is used for fast bufferized i/o operations
  std::stringstream ss(str);
  cell_.records.resize(cell_.num_of_records);
  for (unsigned i = 0; i < cell_.num_of_records; ++i) {
    ss >> cell_.records[i].second_token_id;
    ss >> cell_.records[i].cooc_tf;
    ss >> cell_.records[i].cooc_df;
  }
}

// ******************************* Methods of class ResultingBufferOfCooccurrences ****************************

// ToDo: Test collecting co-occurrences on some small-size exmaples
// ToDo: think about deleting of unordered map from this class, it's unnecessery and all info should be stored in the main class

// The main purpose of this class is to store statistics of co-occurrences and some 
// variables calculated on base of them, perform that calculations, write that in file
// resulting file and read from it. 
// This class stores cells of data from batches before they are written in resulting files
// A cell can from a batch can come in this buffer and be merged with current stored 
// (in case first_token_ids of cells are equal) or the current cell can be pushed from buffer in file 
// (in case first_token_ids aren't equal) and new cell takes place of the old one
ResultingBufferOfCooccurrences::ResultingBufferOfCooccurrences(const int cooc_min_tf, const int cooc_min_df,
    const bool calculate_cooc_tf, const bool calculate_cooc_df,
    const bool calculate_ppmi_tf, const bool calculate_ppmi_df,
    const bool calculate_ppmi, const long long total_num_of_pairs,
    const int total_num_of_documents,
    const std::string& cooc_tf_file_path,
    const std::string& cooc_df_file_path,
    const std::string& ppmi_tf_file_path,
    const std::string& ppmi_df_file_path,
    const std::vector<TokenInfo>& token_statistics_) : cooc_min_tf_(cooc_min_tf),
        cooc_min_df_(cooc_min_df), calculate_cooc_tf_(calculate_cooc_tf),
        calculate_cooc_df_(calculate_cooc_df), calculate_ppmi_tf_(calculate_ppmi_tf),
        calculate_ppmi_df_(calculate_ppmi_df), calculate_ppmi_(calculate_ppmi),
        total_num_of_pairs_(total_num_of_pairs),
        total_num_of_documents_(total_num_of_documents), output_buf_size_(8500),
        open_files_in_buf_(0), token_statistics_(token_statistics_) {

  if (calculate_cooc_tf_) {
    OpenAndCheckOutputFile(cooc_tf_dict_out_, cooc_tf_file_path);
    OpenAndCheckInputFile(cooc_tf_dict_in_, cooc_tf_file_path);
  }

  if (calculate_cooc_df_) {
    OpenAndCheckOutputFile(cooc_df_dict_out_, cooc_df_file_path);
    OpenAndCheckInputFile(cooc_df_dict_in_, cooc_df_file_path);
  }

  if (calculate_ppmi_tf_) {
    OpenAndCheckOutputFile(ppmi_tf_dict_, ppmi_tf_file_path);
  }

  if (calculate_ppmi_df_) {
    OpenAndCheckOutputFile(ppmi_df_dict_, ppmi_df_file_path);
  }
}

void ResultingBufferOfCooccurrences::OpenAndCheckInputFile(std::ifstream& ifile, const std::string& path) {
  ifile.open(path, std::ios::in);
  if (!ifile.good()) {
    throw std::invalid_argument("Failed to create a file in the working directory");
  }
  ++open_files_in_buf_;
}

void ResultingBufferOfCooccurrences::OpenAndCheckOutputFile(std::ofstream& ofile, const std::string& path) {
  ofile.open(path, std::ios::out);
  if (!ofile.good()) {
    throw std::invalid_argument("Failed to create a file in the working directory");
  }
  ++open_files_in_buf_;
}

void ResultingBufferOfCooccurrences::AddInBuffer(const CooccurrenceBatch& batch) {
  if (cell_.first_token_id == batch.cell_.first_token_id) {
    MergeWithExistingCell(batch);
  } else {
    WriteCoocFromBufferInFile();
    cell_ = batch.cell_;
  }
}

void ResultingBufferOfCooccurrences::MergeWithExistingCell(const CooccurrenceBatch& batch) {
  // All the data in buffer are stored in a cell, so here are rules of updating each cell
  // This function takes two vectors (one of the current cell and one which is stored in batch),
  // merges them in folowing way:
  // 1. If two elements of vector are different (different second token id), 
  // stacks them one to another in ascending order
  // 2. It these two elemnts are equal, adds their cooc_tf and cooc_df and 
  // stores resulting cell with this parameters
  // After merging resulting vector is sorted in ascending order
  std::vector<CoocInfo> old_vector = cell_.records;
  cell_.records.resize(old_vector.size() + batch.cell_.records.size());
  auto fi_iter = old_vector.begin();
  auto se_iter = batch.cell_.records.begin();
  auto th_iter = cell_.records.begin();
  while (fi_iter != old_vector.end() && se_iter != batch.cell_.records.end()) {
    if (fi_iter->second_token_id == se_iter->second_token_id) {
      th_iter->second_token_id = fi_iter->second_token_id;
      th_iter->cooc_tf = fi_iter->cooc_tf + se_iter->cooc_tf;
      th_iter->cooc_df = fi_iter->cooc_df + se_iter->cooc_df;
      ++fi_iter;
      ++se_iter;
      ++th_iter;
    } else if (fi_iter->second_token_id < se_iter->second_token_id) {
      *(th_iter++) = *(fi_iter++);
    } else {
      *(th_iter++) = *(se_iter++);
    }
  }
  cell_.records.resize(th_iter - cell_.records.begin());
  std::copy(fi_iter, old_vector.end(), std::back_inserter(cell_.records));
  std::copy(se_iter, batch.cell_.records.end(), std::back_inserter(cell_.records));
}

// ToDo: continue here

// Output file format (of variety of formats) is difined here
void ResultingBufferOfCooccurrences::WriteCoocFromBufferInFile() {
  // This function takes the cell of the buffer, writes data from cell in file,
  // performing calculation of number of n_u (number of pairs where a specific 
  // token u co-occurred with any token) in a window of specified width - this
  // information will needed for ppmi computation
  // So this function perform 2 different things - writing data in file
  // and calculation of some variable needed in future ppmi, but it's
  // more optimal to compute this variable while walking through co-occurence statistics
  // then to do it in another moment separately

  // stringstream is used for fast bufferized i/o operations
  std::stringstream output_buf_tf;
  std::stringstream output_buf_df;

  // Calculate statistics of occurrence for a new token (as a first token in some pair)
  PpmiCountersValues n_u;
  for (unsigned i = 0; i < cell_.records.size(); ++i) {
    if (calculate_cooc_tf_ && cell_.records[i].cooc_tf >= cooc_min_tf_) {
      // Cooccurrence of the same tokens isn't interested for us
      if (cell_.first_token_id != cell_.records[i].second_token_id) {
        output_buf_tf << cell_.first_token_id << ' ' << cell_.records[i].second_token_id 
                                              << ' ' << cell_.records[i].cooc_tf << std::endl;
      }
      // That's how counter n_u used in ppmi formula is computed
      n_u.n_u_tf += cell_.records[i].cooc_tf;
    }
    if (output_buf_tf.tellg() > output_buf_size_) {
      cooc_tf_dict_out_ << output_buf_tf.str();
    }

    if (calculate_cooc_df_ && cell_.records[i].cooc_df >= cooc_min_df_) {
      // Cooccurrence of the same tokens isn't interested for us
      if (cell_.first_token_id != cell_.records[i].second_token_id) {
        output_buf_df << cell_.first_token_id << ' ' << cell_.records[i].second_token_id
                                              << ' ' << cell_.records[i].cooc_df << std::endl;
      }
    }
    if (output_buf_df.tellg() > output_buf_size_) {
      cooc_df_dict_out_ << output_buf_df.str();
    }
  }

  if (calculate_cooc_tf_) {
    cooc_tf_dict_out_ << output_buf_tf.str();
  }
  if (calculate_cooc_df_) {
    cooc_df_dict_out_ << output_buf_df.str();
  }

  // Save calculated value in hash table of ppmi counters (because it's needed for ppmi calculation)
  ppmi_counters_.insert(std::make_pair(cell_.first_token_id, n_u));
  // It's importants after pop to set size = 0, because this value will be checked later
  cell_.records.resize(0);
}

// ToDo: erase duplications of code
void ResultingBufferOfCooccurrences::CalculateAndWritePpmi() {
  // stringstream is used for fast bufferized i/o operations
  std::stringstream output_buf_tf;
  std::stringstream output_buf_df;
  int first_token_id = 0;
  int second_token_id = 0;
  long long cooc_tf = 0;
  int cooc_df = 0;
  if (calculate_ppmi_tf_) {
    while (cooc_tf_dict_in_ >> first_token_id) {
      cooc_tf_dict_in_ >> second_token_id;
      cooc_tf_dict_in_ >> cooc_tf;
      if (first_token_id > second_token_id) {
        continue;
      }
      double sub_log_tf_pmi = (static_cast<double>(total_num_of_pairs_) /
          ppmi_counters_[first_token_id].n_u_tf) /
          (ppmi_counters_[second_token_id].n_u_tf / static_cast<double>(cooc_tf));
      if (sub_log_tf_pmi > 1.0) {
        output_buf_tf << first_token_id << ' ' << second_token_id << ' ' << log(sub_log_tf_pmi) << std::endl;
        if (output_buf_tf.tellg() > output_buf_size_) {
          ppmi_tf_dict_ << output_buf_tf.str();
        }
      }
    }
    ppmi_tf_dict_ << output_buf_tf.str();
  }
  if (calculate_ppmi_df_) {
    while (cooc_df_dict_in_ >> first_token_id) {
      cooc_df_dict_in_ >> second_token_id;
      cooc_df_dict_in_ >> cooc_df;
      if (first_token_id > second_token_id) {
        continue;
      }
      double sub_log_df_pmi = (static_cast<double>(total_num_of_documents_) /
          ppmi_counters_[first_token_id].n_u_df /*token_statistics_[first_token_id].num_of_documents_token_occured_in*/) /
          (ppmi_counters_[second_token_id].n_u_df /*token_statistics_[second_token_id].num_of_documents_token_occured_in*/ / static_cast<double>(cooc_df));
      if (sub_log_df_pmi > 1.0) {
        output_buf_df << first_token_id << ' ' << second_token_id << ' ' << log(sub_log_df_pmi) << std::endl;
        if (output_buf_df.tellg() > output_buf_size_) {
          ppmi_df_dict_ << output_buf_df.str();
        }
      }
    }
    ppmi_df_dict_ << output_buf_df.str();
  }
}
