// Copyright 2018, Additive Regularization of Topic Models.

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
#include <chrono>
#include <thread>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/utility.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

namespace fs = boost::filesystem;

// ****************************** Methods of class CooccurrenceDictionary ***********************************

CooccurrenceDictionary::CooccurrenceDictionary(const unsigned window_width,
    const unsigned cooc_min_tf, const unsigned cooc_min_df,
    const std::string& path_to_vocab, const std::string& path_to_vw,
    const std::string& cooc_tf_file_path,
    const std::string& cooc_df_file_path,
    const std::string& ppmi_tf_file_path,
    const std::string& ppmi_df_file_path, const int num_of_cpu,
    const unsigned doc_per_cooc_batch) :
        window_width_(window_width), cooc_min_tf_(cooc_min_tf), cooc_min_df_(cooc_min_df), 
        path_to_vw_(path_to_vw),
        cooc_tf_file_path_(cooc_tf_file_path),
        cooc_df_file_path_(cooc_df_file_path),
        ppmi_tf_file_path_(ppmi_tf_file_path),
        ppmi_df_file_path_(ppmi_df_file_path),
        vocab_(path_to_vocab),
        open_files_counter_(0), max_num_of_open_files_(500), total_num_of_pairs_(0),
        total_num_of_documents_(0), doc_per_cooc_batch_(doc_per_cooc_batch) {
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
  token_statistics_.resize(vocab_.storage_.size());
  path_to_batches_ = dir.string();
  if (num_of_cpu <= 0) {
    num_of_cpu_ = std::thread::hardware_concurrency();
    if (num_of_cpu_ == 0) {
      num_of_cpu_ = 1;
    }
  } else {
    num_of_cpu_ = num_of_cpu;
  }
  std::cout << "documents per batch = " << doc_per_cooc_batch_ << std::endl;
}

unsigned CooccurrenceDictionary::VocabSize() const {
  return vocab_.storage_.size();
}

void CooccurrenceDictionary::ReadVowpalWabbit() {
  // This function works as follows:
  // 1. Acquire lock for reading from vowpal wabbit file
  // 2. Read a portion (doc_per_cooc_batch) of documents from file and save it
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
    throw std::invalid_argument("Failed to open vowpal wabbit file");
  }
  std::mutex read_lock;
  std::mutex stdout_lock;
  std::mutex token_statistics_arg_access_lock;
  std::mutex total_num_of_documents_arg_access_lock;
  std::mutex total_num_of_pairs_arg_access_lock;

  auto func = [&]() {
    while (true) { // Loop throgh portions.
      // Steps 1-3:
      std::vector<std::string> portion = ReadPortionOfDocuments(read_lock, vowpal_wabbit_doc);
      if (portion.empty()) {
        break;
      }
      {
        std::lock_guard<std::mutex> guard(total_num_of_documents_arg_access_lock);
        total_num_of_documents_ += portion.size(); // statistics for ppmi
      }
      // It will hold tf and df of pairs of tokens
      // Every pair of valid tokens (both exist in vocab) is saved in this storage
      // After walking through portion of documents all the statistics is dumped on disk
      // and then this storage is destroyed
      CooccurrenceStatisticsHolder cooc_stat_holder;

      // For every token from vocab keep the information about the last document this token occured in
      std::vector<unsigned> num_of_last_document_token_occured(vocab_.storage_.size());

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
          int first_token_id = vocab_.FindToken(doc[j], "@default_class");
          if (first_token_id == TOKEN_NOT_FOUND) {
            continue;
          }
          // 5.c) Collect statistics for document ppmi (in how many documents every token occured)
          // The array is initialized with zeros, so for every portion.size() it isn't equal to initial value
          if (num_of_last_document_token_occured[first_token_id] != portion.size()) {
            num_of_last_document_token_occured[first_token_id] = portion.size();
            std::lock_guard<std::mutex> guard(token_statistics_arg_access_lock);
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
            int second_token_id = vocab_.FindToken(doc[j + k], "@default_class");
            if (second_token_id == TOKEN_NOT_FOUND) {
              continue;
            }
            // 5.e) When it's known these 2 tokens are valid, remember their co-occurrence
            // Here portion.size() is used to identify a document (it's unique id during one portion of documents)
            cooc_stat_holder.SavePairOfTokens(first_token_id, second_token_id, portion.size());
            cooc_stat_holder.SavePairOfTokens(second_token_id, first_token_id, portion.size());
            std::lock_guard<std::mutex> guard(total_num_of_pairs_arg_access_lock);
            total_num_of_pairs_ += 2; // statistics for ppmi
          }
        }
      }
      if (!cooc_stat_holder.storage_.empty()) {
        // This function saves gathered statistics on disk
        // After saving on disk statistics from all the batches needs to be merged
        // This is implemented in ReadAndMergeCooccurrenceBatches(), so the next step is to call this method
        // Sorting is needed before storing all pairs of tokens on disk (it's for future agregation)
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
  for (unsigned i = 0; i < num_of_cpu_; ++i) {
    tasks.emplace_back(std::async(std::launch::async, func));
  }
  for (unsigned i = 0; i < num_of_cpu_; ++i) {
    tasks[i].get();
  }
  std::cout << '\n' << "Co-occurrence batches have been created" << std::endl;
}

std::vector<std::string> CooccurrenceDictionary::ReadPortionOfDocuments(
              std::mutex& read_lock, std::ifstream& vowpal_wabbit_doc) {
  std::vector<std::string> portion;
  std::lock_guard<std::mutex> guard(read_lock);
  if (vowpal_wabbit_doc.eof()) {
    return portion;
  }
  std::string str;
  while (portion.size() < doc_per_cooc_batch_) {
    getline(vowpal_wabbit_doc, str);
    if (vowpal_wabbit_doc.eof()) {
      break;
    }
    portion.push_back(str);
  }
  return portion;
}

int CooccurrenceDictionary::SetModalityLabel(const std::string& modality_label) const {
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

CooccurrenceBatch* CooccurrenceDictionary::CreateNewCooccurrenceBatch() const {
  return new CooccurrenceBatch(path_to_batches_);
}

void CooccurrenceDictionary::OpenBatchOutputFile(CooccurrenceBatch& batch) {
  if (!batch.out_batch_.is_open()) {
    assert(open_files_counter_ < max_num_of_open_files_);
    ++open_files_counter_;
    batch.out_batch_.open(batch.filename_, std::ios::out);
  }
}

void CooccurrenceDictionary::CloseBatchOutputFile(CooccurrenceBatch& batch) {
  if (batch.out_batch_.is_open()) {
    --open_files_counter_;
    batch.out_batch_.close();
  }
}

unsigned CooccurrenceDictionary::CooccurrenceBatchesQuantity() const {
  return vector_of_batches_.size();
}

// ToDo: split files that can be open now into n groups
// When a thread has eliberated, take one more group
// Recursively agregate all the files (it won't need reopenning)
/*ResultingBufferOfCooccurrences CooccurrenceDictionary::ReadAndMergeCooccurrenceBatches() {
  // After that all the statistics has been gathered and saved in form of cooc batches on disk, it
  // needs to be read and merged from cooc batches into one storage
  // (This is may be the longest-time part of co-occurrence gathering).
  // All the co-occurrence batches are split equally into n groups, where n = number of cores.
  // There are two stages of merging:
  // 1. merging of files of one group (is done asynchroniously, without dropping of rare pairs of tokens)
  // n files are written back in the same form (of cooccurrence batches)
  // 2. then that n files need to be read and merged again (with dropping of rare pairs of tokens)
  // Merging of k files is implemented in KWayMerge function
  // After the second stage of merging data is written in format of output file (not in format of cooc batches)
  // If there would be a need to calculate ppmi or other values which depend on co-occurrences
  // this data can be read back from output file.
  // Also there's one more possible strategy of paralleling: n sells can be simultaniously
  // asynchroniously gathered, then merged using 1 thread and pushed in output file
  // Pro is that that's not necessery to collect some intermediate batches
  // Cons are: time of argegation depends on split, all the threads should wait for the slowest thread
  std::cout << "Step 2: merging batches" << std::endl;
  // Stage 1: creation of intermediate batches
  unsigned num_of_threads = std::min(static_cast<unsigned>(vector_of_batches_.size()), num_of_cpu_);
  //num_of_threads = 1; // ToDo:
  std::mutex open_close_file_lock;
  std::vector<std::unique_ptr<CooccurrenceBatch>> intermediate_batches;
  if (num_of_threads != 1) { // multithread implementation of reading and merging of batches
    for (unsigned i = 0; i < num_of_threads; ++i) { // Create intermediate batches
      std::unique_ptr<CooccurrenceBatch> batch(CreateNewCooccurrenceBatch());
      OpenBatchOutputFile(*batch);
      intermediate_batches.push_back(std::move(batch));
    }
    // It's better if in at least one thread all the batches are open than
    // in all the batches is at least one closed batch
    for (unsigned i = 0; open_files_counter_ < max_num_of_open_files_ - 1 &&
                                           i < vector_of_batches_.size(); ++i) {
      OpenBatchInputFile(*vector_of_batches_[i]);
    }
    auto func = [&](unsigned i) { // Wrapper around KWayMerge (stage 1)
      // Split all batches into groups
      std::vector<std::unique_ptr<CooccurrenceBatch>> part_of_batches;
      for (unsigned j = (i * vector_of_batches_.size()) / num_of_threads;
              j < ((i + 1) * vector_of_batches_.size()) / num_of_threads; ++j) {
        part_of_batches.push_back(std::move(vector_of_batches_[j]));
      }
      ResultingBufferOfCooccurrences intermediate_buffer(token_statistics_);
      KWayMerge(intermediate_buffer, BATCH, part_of_batches, *intermediate_batches[i], open_close_file_lock);
      CloseBatchOutputFile(*intermediate_batches[i]);
    };
    // Stage 1: async launch
    std::vector<std::shared_future<void>> tasks;
    for (unsigned i = 0; i < num_of_threads; ++i) {
      tasks.emplace_back(std::async(std::launch::async, func, i));
    }
    for (unsigned i = 0; i < num_of_threads; ++i) {
      tasks[i].get();
    }
  }
  // Stage 2: merging of final batches (in single thread)
  ResultingBufferOfCooccurrences res(token_statistics_, cooc_min_tf_, cooc_min_df_,
                                     total_num_of_pairs_, total_num_of_documents_,
                                     calculate_tf_cooc_, calculate_df_cooc_,
                                     calculate_tf_ppmi_, calculate_df_ppmi_, calculate_ppmi_,
                                     cooc_tf_file_path_, cooc_df_file_path_,
                                     ppmi_tf_file_path_, ppmi_df_file_path_);
  open_files_counter_ += res.open_files_in_buf_;
  std::mutex open_close_file_lock; // ToDo:
  if (num_of_threads == 1) {
    // ToDo: replace 4th fake arg (how if they're passed be reference?)
    KWayMerge(res, OUTPUT_FILE, vector_of_batches_, *vector_of_batches_[0], open_close_file_lock);
  } else {
    KWayMerge(res, OUTPUT_FILE, intermediate_batches, *intermediate_batches[0], open_close_file_lock);
  }
  // Files are explicitly closed here, because it's necesery to push the data in files on this step
  if (calculate_tf_cooc_) {
    res.cooc_tf_dict_out_.close();
  }
  if (calculate_df_cooc_) {
    res.cooc_df_dict_out_.close();
  }
  open_files_counter_ -= 2;
  std::cout << "Batches have been merged" << std::endl;
  return res;
}*/

ResultingBufferOfCooccurrences CooccurrenceDictionary::ReadAndMergeCooccurrenceBatches() {
  std::cout << "Step 2: merging batches" << std::endl;
  ResultingBufferOfCooccurrences res(token_statistics_, cooc_min_tf_, cooc_min_df_,
                                     total_num_of_pairs_, total_num_of_documents_,
                                     calculate_tf_cooc_, calculate_df_cooc_,
                                     calculate_tf_ppmi_, calculate_df_ppmi_, calculate_ppmi_,
                                     cooc_tf_file_path_, cooc_df_file_path_,
                                     ppmi_tf_file_path_, ppmi_df_file_path_);
  open_files_counter_ += res.open_files_in_buf_;
  std::mutex open_close_file_lock;
  KWayMerge(res, OUTPUT_FILE, vector_of_batches_, *vector_of_batches_[0], open_close_file_lock);
  if (calculate_tf_cooc_) {
    res.cooc_tf_dict_out_.close();
  }
  if (calculate_df_cooc_) {
    res.cooc_df_dict_out_.close();
  }
  open_files_counter_ -= 2;
  std::cout << "Batches have been merged" << std::endl;
  return res;
}

void CooccurrenceDictionary::KWayMerge(ResultingBufferOfCooccurrences& res, const int mode,
                                       std::vector<std::unique_ptr<CooccurrenceBatch>>& vector_of_input_batches,
                                       CooccurrenceBatch& out_batch, std::mutex& open_close_file_lock) {
  // All cooc batches has it's local buffer in operating memory (look the CooccurrenceBatch class implementation)
  // Information in batches is stored in cells.
  // There are 2 different output formats, which are set via mode parameter:
  // 1. Batches
  // 2. Output file
  // Here's the k-way merge algorithm for external sorting:
  // 1. Initially first cells of all the batches are read in their buffers
  // 2. Then batches are sorted (std::make_heap) by first_token_id of the cell
  // 3. Then a cell with the lowest first_token_id is extaracted and put in
  // resulting buffer and the next cell is read from corresponding batch
  // 4. If the lowest first token id equals first token id of cell of this buffer, they are merged
  // else the current cell is written in file and the new one is loaded.
  // Writing and empting is done in order to keep low memory consumption
  // During execution of this function (if mode is OUTPUT_FILE) n_u is calculated and saved,
  // so after merge all the information needed to calcualte ppmi will be gathered and
  // available from class ResultingBufferOfCooccurrences
  // Note: here's only 1 way to communicate between threads - through open_files_counter

  // Step 1:
  auto iter = vector_of_input_batches.begin();
  {
    std::lock_guard<std::mutex> guard(open_close_file_lock);
    for (; iter != vector_of_input_batches.end() && open_files_counter_ < max_num_of_open_files_ - 1; ++iter) {
      OpenBatchInputFile(**iter);
      (*iter)->ReadCell();
    }
  }
  for (; iter != vector_of_input_batches.end(); ++iter) {
    std::lock_guard<std::mutex> guard(open_close_file_lock);
    OpenBatchInputFile(**iter);
    (*iter)->ReadCell();
    CloseBatchInputFile(**iter);
  }
  // Step 2:
  std::make_heap(vector_of_input_batches.begin(), vector_of_input_batches.end(),
                 CooccurrenceBatch::CoocBatchComparator());
  if (!vector_of_input_batches.empty()) {
    res.cell_ = Cell();
    res.cell_.first_token_id = (*vector_of_input_batches[0]).cell_.first_token_id;
  }
  while (!vector_of_input_batches.empty()) {
    // Step 4:
    if (res.cell_.first_token_id == (*vector_of_input_batches[0]).cell_.first_token_id) {
      res.MergeWithExistingCell(*vector_of_input_batches[0]);
    } else {
      if (mode == BATCH) {
        out_batch.cell_ = res.cell_;
        out_batch.WriteCell();
      } else if (mode == OUTPUT_FILE) {
        res.WriteCoocFromCellInFile(); // This function also performs n_u calculating
      }
      res.cell_ = (*vector_of_input_batches[0]).cell_;
    }
    // Step 3:
    std::pop_heap(vector_of_input_batches.begin(), vector_of_input_batches.end(),
                  CooccurrenceBatch::CoocBatchComparator());
    if (!vector_of_input_batches.back()->in_batch_.is_open()) {
      std::lock_guard<std::mutex> guard(open_close_file_lock);
      OpenBatchInputFile(*vector_of_input_batches.back());
    }
    // if there are some data to read ReadCell reads it and returns true, else returns false
    if (vector_of_input_batches.back()->ReadCell()) {
      if (max_num_of_open_files_ == open_files_counter_) {
        std::lock_guard<std::mutex> guard(open_close_file_lock);
        CloseBatchInputFile(*vector_of_input_batches.back());
      }
      std::push_heap(vector_of_input_batches.begin(), vector_of_input_batches.end(),
                     CooccurrenceBatch::CoocBatchComparator());
    } else {
      if (IsOpenBatchInputFile(*vector_of_input_batches.back())) {
        std::lock_guard<std::mutex> guard(open_close_file_lock);
        CloseBatchInputFile(*vector_of_input_batches.back());
      }
      vector_of_input_batches.pop_back();
    }
  }
  if (!res.cell_.records.empty()) {
    if (mode == BATCH) {
      out_batch.cell_ = res.cell_;
      out_batch.WriteCell();
    } else if (mode == OUTPUT_FILE) {
      res.WriteCoocFromCellInFile(); // This function also performs n_u calculating
    }
  }
}

bool CooccurrenceDictionary::GetCalculatePpmi() const {
  return calculate_ppmi_;
}

void CooccurrenceDictionary::OpenBatchInputFile(CooccurrenceBatch& batch) {
  assert(open_files_counter_ < max_num_of_open_files_);
  if (!batch.in_batch_.is_open()) {
    ++open_files_counter_;
    batch.in_batch_.open(batch.filename_, std::ios::in);
    batch.in_batch_.seekg(batch.in_batch_offset_);
  }
}

bool CooccurrenceDictionary::IsOpenBatchInputFile(const CooccurrenceBatch& batch) const {
  return batch.in_batch_.is_open();
}

void CooccurrenceDictionary::CloseBatchInputFile(CooccurrenceBatch& batch) {
  if (batch.in_batch_.is_open()) {
    --open_files_counter_;
    batch.in_batch_offset_ = batch.in_batch_.tellg();
    batch.in_batch_.close();
  }
}

CooccurrenceDictionary::~CooccurrenceDictionary() {
  fs::remove_all(path_to_batches_);
}

// ********************************** Methods of class Vocab *****************************************

Vocab::Vocab(const std::string& path_to_vocab) {
  // This function constructs vocab object: reads tokens from vocab file,
  // sets them unique id and collects pairs in dictionary
  std::ifstream vocab_ifile(path_to_vocab, std::ios::in);
  if (!vocab_ifile.good()) {
    throw std::invalid_argument("Failed to open vocab file, path = " + path_to_vocab);
  }
  std::string str;
  for (unsigned last_token_id = 0; getline(vocab_ifile, str); ++last_token_id) {
    boost::algorithm::trim(str);
    std::vector<std::string> strs;
    boost::split(strs, str, boost::is_any_of(" "));
    if (!strs[0].empty()) {
      std::string modality;
      if (strs.size() == 1) {
        modality = "@default_class";
      } else {
        modality = strs[1];
      }
      std::string key = MakeKey(strs[0], modality);
      auto iter = storage_.find(key);
      if (iter == storage_.end()) {
        storage_.insert(std::make_pair(key, last_token_id));
      } else {
        throw std::invalid_argument("There repeated tokens in vocab file. Please remove all the duplications");
      }
    }
  }
}

std::string Vocab::MakeKey(const std::string& token_str, const std::string& modality) const {
  return token_str + '|' + modality;
}

int Vocab::FindToken(const std::string& token_str, const std::string& modality) const {
  std::string key = MakeKey(token_str, modality);
  auto token_ref = storage_.find(key);
  if (token_ref == storage_.end()) {
    return TOKEN_NOT_FOUND;
  }
  return token_ref->second; // Returns token_id
}

// ****************************** Methods of class CooccurrenceStatisticsHolder *************************************

// This class stores temporarily added statistics about pairs of tokens (how often these pairs occurred in documents
// in a window and in how many documents they occurred together in a window).
// The data is stored in rb tree (std::map)
void CooccurrenceStatisticsHolder::SavePairOfTokens(const int first_token_id, const int second_token_id,
                                                    const unsigned doc_id) {
  // There are 2 levels of indexing
  // The first level keeps information about first token and the second level
  // about co-occurrence between the first and the second tokens
  // If first token id is known (exists in the structure), corresponding node should be modified
  // else it should be added to the structure
  auto it1 = storage_.find(first_token_id);
  if (it1 == storage_.end()) {
    FirstToken first_token;
    first_token.second_token_reference = std::map<int, SecondTokenAndCooccurrence> {
      {second_token_id, SecondTokenAndCooccurrence(doc_id)}
    };
    storage_.insert(std::pair<int, FirstToken>(first_token_id, first_token));
  } else {
    std::map<int, SecondTokenAndCooccurrence>& second_tokens = it1->second.second_token_reference;
    auto it2 = second_tokens.find(second_token_id);
    if (it2 == second_tokens.end()) {
      SecondTokenAndCooccurrence second_token(doc_id);
      second_tokens.insert(std::pair<int, SecondTokenAndCooccurrence>(second_token_id, second_token));
    } else {
      if (it2->second.last_doc_id != doc_id) {
        it2->second.last_doc_id = doc_id;
        ++(it2->second.cooc_df);
      }
      ++(it2->second.cooc_tf);
    }
  }
}

// **************************************** Methods of class CooccurrenceBatch ********************************************

CooccurrenceBatch::CooccurrenceBatch(const std::string& path_to_batches) : in_batch_offset_(0) {
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  fs::path batch(boost::lexical_cast<std::string>(uuid));
  fs::path full_filename = fs::path(path_to_batches) / batch;
  filename_ = full_filename.string();
}

void CooccurrenceBatch::FormNewCell(const std::map<int, CooccurrenceStatisticsHolder::FirstToken>::iterator& cooc_stat_node) {
  // Here is initialization of a new cell
  // A cell consists on first_token_id, number of records it includes
  // Then records go, every reord consists on second_token_id, cooc_tf, cooc_df

  cell_.first_token_id = cooc_stat_node->first;
  // while reading from file it's necessery to know how many records to read
  std::map<int, CooccurrenceStatisticsHolder::
                SecondTokenAndCooccurrence>& second_token_reference = cooc_stat_node->second.second_token_reference;
  cell_.num_of_records = second_token_reference.size();
  cell_.records.resize(cell_.num_of_records);
  unsigned i = 0;
  for (auto iter = second_token_reference.begin(); iter != second_token_reference.end(); ++iter, ++i) {
    cell_.records[i].second_token_id = iter->first;
    cell_.records[i].cooc_tf = iter->second.cooc_tf;
    cell_.records[i].cooc_df = iter->second.cooc_df;
  }
}

void CooccurrenceBatch::WriteCell() {
  // Cells are written in following form: first line consists of first token id and num of triples
  // the second line consists of numbers triples, which are separeted with a
  // space and numbers in these triples are separeted the same
  // stringstream is used for fast bufferized i/o operations
  // Initially data is written into stringstreams and then one large stringstream is written in file
  std::stringstream ss;
  ss << cell_.first_token_id << ' ';
  // The value of the variable cell_.num_of_records is invalid (wasn't updated from first read from batch)
  ss << cell_.records.size() << std::endl;
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

// **************************** Methods of class ResultingBufferOfCooccurrences ********************************

// The main purpose of this class is to store statistics of co-occurrences and some 
// variables calculated on base of them, perform that calculations, write that in file
// resulting file and read from it. 
// This class stores cells of data from batches before they are written in resulting files
// A cell can from a batch can come in this buffer and be merged with current stored 
// (in case first_token_ids of cells are equal) or the current cell can be pushed from buffer in file 
// (in case first_token_ids aren't equal) and new cell takes place of the old one
ResultingBufferOfCooccurrences::ResultingBufferOfCooccurrences(
    std::vector<TokenInfo>& token_statistics_,
    const unsigned cooc_min_tf, const unsigned cooc_min_df,
    const unsigned long long total_num_of_pairs, const unsigned total_num_of_documents,
    const bool calculate_cooc_tf, const bool calculate_cooc_df,
    const bool calculate_ppmi_tf, const bool calculate_ppmi_df,
    const bool calculate_ppmi,
    const std::string& cooc_tf_file_path, const std::string& cooc_df_file_path,
    const std::string& ppmi_tf_file_path, const std::string& ppmi_df_file_path) :
        token_statistics_(token_statistics_),
        cooc_min_tf_(cooc_min_tf), cooc_min_df_(cooc_min_df),
        total_num_of_pairs_(total_num_of_pairs),
        total_num_of_documents_(total_num_of_documents),
        output_buf_size_(8500), open_files_in_buf_(0),
        calculate_cooc_tf_(calculate_cooc_tf), calculate_cooc_df_(calculate_cooc_df),
        calculate_ppmi_tf_(calculate_ppmi_tf), calculate_ppmi_df_(calculate_ppmi_df),
        calculate_ppmi_(calculate_ppmi) {
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
    throw std::invalid_argument("Failed to create a file in working directory");
  }
  ++open_files_in_buf_;
}

void ResultingBufferOfCooccurrences::OpenAndCheckOutputFile(std::ofstream& ofile, const std::string& path) {
  ofile.open(path, std::ios::out);
  if (!ofile.good()) {
    throw std::invalid_argument("Failed to create a file in working directory");
  }
  ++open_files_in_buf_;
}

// ToDo: may be this can be implemented in more optimal way
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
      *th_iter++ = *fi_iter++;
    } else {
      *th_iter++ = *se_iter++;
    }
  }
  cell_.records.resize(th_iter - cell_.records.begin());
  std::copy(fi_iter, old_vector.end(), std::back_inserter(cell_.records));
  std::copy(se_iter, batch.cell_.records.end(), std::back_inserter(cell_.records));
}

void ResultingBufferOfCooccurrences::WriteCoocFromCellInFile() { // Output file formats are defined here
  // This function takes a cell from buffer and writes data from cell in file
  // and at the same time it calculates n_u (which is number of pairs where a 
  // token u which is assosiated with the given cell co-occurred with any token 
  // in a window of specified width. This information will be needed for ppmi
  // computation. And this exact value is returned from the function (it goes in
  // token_statistics_ variable from class CooccurrenceDictionary)
  // So this function performs 2 different things - writing data in file
  // and calculation of some variable needed in future ppmi
  // It might seem strange, but it's more optimal to compute this variable while 
  // walking through data which is going to be dumped in file right now, than to perform
  // double walking through the data

  // stringstream is used for fast bufferized i/o operations
  // Initially data is written into stringstreams and then one large stringstream is written in file 
  std::stringstream output_buf_tf;
  std::stringstream output_buf_df;

  // Calculate statistics of occurrence for a new token (as a first token in some pair)
  unsigned long long n_u = 0;
  for (unsigned i = 0; i < cell_.records.size(); ++i) {
    if (calculate_cooc_tf_ && cell_.records[i].cooc_tf >= cooc_min_tf_) {
      // Co-occurrence of the same tokens isn't interested for us
      if (cell_.first_token_id != cell_.records[i].second_token_id) {
        output_buf_tf << cell_.first_token_id << ' ' << cell_.records[i].second_token_id
                                              << ' ' << cell_.records[i].cooc_tf << std::endl;
      }
      // That's how counter n_u (which is used in ppmi formula) is calculated
      n_u += cell_.records[i].cooc_tf;
    }
    if (output_buf_tf.tellg() > output_buf_size_) {
      cooc_tf_dict_out_ << output_buf_tf.str();
    }

    if (calculate_cooc_df_ && cell_.records[i].cooc_df >= cooc_min_df_) {
      // Co-occurrence of the same tokens isn't interested for us
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
  // It's importants to set size = 0 after popping, because this value will be checked later
  cell_.records.resize(0);
  token_statistics_[cell_.first_token_id].num_of_pairs_token_occured_in = n_u;
}

// ToDo: erase duplications of code
// ToDo: Write parallel implementation
void ResultingBufferOfCooccurrences::CalculateAndWritePpmi() {
  // This function reads cooc file line by line, calcualtes ppmi and saves them in external file of ppmi
  /*{ // ToDo: insert this in test file
    std::ofstream out("token_stat_", std::ios::out);
    out << "total_num_of_pairs = " << total_num_of_pairs_ << ", total_num_of_documents = "
                                   << total_num_of_documents_ << '\n' << "token_id, n_u_t, n_u_d\n";
    for (unsigned i = 0; i < token_statistics_.size(); ++i) {
      out << i << ' ' << token_statistics_[i].num_of_pairs_token_occured_in <<
                  ' ' << token_statistics_[i].num_of_documents_token_occured_in << '\n';
    }
  }*/
  std::cout << "Step 3: start calculation ppmi" << std::endl;
  // stringstream is used for fast bufferized i/o operations
  std::stringstream output_buf_tf;
  std::stringstream output_buf_df;
  int first_token_id = 0;
  int second_token_id = 0;
  unsigned long long cooc_tf = 0;
  unsigned cooc_df = 0;
  if (calculate_ppmi_tf_) {
    while (cooc_tf_dict_in_ >> first_token_id) {
      cooc_tf_dict_in_ >> second_token_id;
      cooc_tf_dict_in_ >> cooc_tf;
      if (first_token_id > second_token_id) {
        continue;
      }
      double n_u = token_statistics_[first_token_id].num_of_pairs_token_occured_in;
      double n_v = token_statistics_[second_token_id].num_of_pairs_token_occured_in;
      double n = static_cast<long double>(total_num_of_pairs_);
      double n_uv = static_cast<double>(cooc_tf);
      double under_log_tf_pmi = (n / n_u) / (n_v / n_uv);
      if (under_log_tf_pmi > 1.0) {
        output_buf_tf << first_token_id << ' ' << second_token_id << ' ' << log(under_log_tf_pmi) << std::endl;
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
      double N_u = token_statistics_[first_token_id].num_of_documents_token_occured_in;
      double N_v = token_statistics_[second_token_id].num_of_documents_token_occured_in;
      double N = static_cast<double>(total_num_of_documents_);
      double N_uv = static_cast<double>(cooc_df);
      double under_log_df_pmi = (N / N_u) / (N_v / N_uv);
      if (under_log_df_pmi > 1.0) {
        output_buf_df << first_token_id << ' ' << second_token_id << ' ' << log(under_log_df_pmi) << std::endl;
        if (output_buf_df.tellg() > output_buf_size_) {
          ppmi_df_dict_ << output_buf_df.str();
        }
      }
    }
    ppmi_df_dict_ << output_buf_df.str();
  }
  std::cout << "Ppmi's have been calculated" << std::endl;
}
