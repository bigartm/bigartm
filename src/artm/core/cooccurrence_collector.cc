// Copyright 2018, Additive Regularization of Topic Models.

#include "artm/core/cooccurrence_collector.h"

#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <future>  // NOLINT
#include <mutex>  // NOLINT
#include <thread>  // NOLINT
#include <sstream>
#include <iomanip>
#include <memory>
#include <cassert>
#include <stdexcept>
#include <queue>
#include <algorithm>
#include <utility>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/utility.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

#include "artm/core/common.h"
#include "artm/core/exceptions.h"
#include "artm/core/collection_parser.h"

namespace fs = boost::filesystem;

namespace artm {
namespace core {

// ToDo (MichaelSolotky): add this input format to (batches, (optionally) Dictionary)
// 1. find a point in code after collection parsing where this code can be inserted
// 2. take parser config from Parse method and find batch folder name
// 3. use method ListAllBatches to have vector of filenames
// 4. read batches in some order and count co-occurrences (use method LoadBatches)
// 4.a understand how to extract text from batches
// 5. think how to specify ppmi and cooc output files

// ToDo (MichaelSolotky): write docs
// ToDo (MichaelSolotky): search for all bad-written parts of code with CLion

// ****************************** Methods of class CooccurrenceCollector ***********************************

CooccurrenceCollector::CooccurrenceCollector(const CollectionParserConfig& collection_parser_config) {
  config_.set_gather_cooc(collection_parser_config.gather_cooc());
  if (config_.gather_cooc()) {
    config_.set_gather_cooc_tf(collection_parser_config.gather_cooc_tf());
    config_.set_gather_cooc_df(collection_parser_config.gather_cooc_df());
    config_.set_use_symetric_cooc(true);
    config_.set_vw_file_path(collection_parser_config.docword_file_path());

    if (collection_parser_config.has_vocab_file_path()) {
      config_.set_vocab_file_path(collection_parser_config.vocab_file_path());
      vocab_ = Vocab(config_.vocab_file_path());
      num_of_documents_token_occurred_in_.resize(vocab_.token_map_.size());
    } else {
      BOOST_THROW_EXCEPTION(InvalidOperation("No vocab file specified. Can't gather co-occurrences"));
    }
    config_.set_target_folder(collection_parser_config.target_folder());

    if (collection_parser_config.has_cooc_tf_file_path()) {
      config_.set_cooc_tf_file_path(collection_parser_config.cooc_tf_file_path());
    } else if (config_.gather_cooc_tf()) {
      config_.set_cooc_tf_file_path(CreateFileInBatchDir());
    }

    if (collection_parser_config.has_cooc_df_file_path()) {
      config_.set_cooc_df_file_path(collection_parser_config.cooc_df_file_path());
    } else if (config_.gather_cooc_df()) {
      config_.set_cooc_df_file_path(CreateFileInBatchDir());
    }

    if (collection_parser_config.has_ppmi_tf_file_path()) {
      config_.set_ppmi_tf_file_path(collection_parser_config.ppmi_tf_file_path());
      config_.set_calculate_ppmi_tf(true);
    } else {
      config_.set_calculate_ppmi_tf(false);
    }
    if (collection_parser_config.has_ppmi_df_file_path()) {
      config_.set_ppmi_df_file_path(collection_parser_config.ppmi_df_file_path());
      config_.set_calculate_ppmi_df(true);
    } else {
      config_.set_calculate_ppmi_df(false);
    }

    config_.set_cooc_window_width(collection_parser_config.cooc_window_width());
    config_.set_cooc_min_tf(collection_parser_config.cooc_min_tf());
    config_.set_cooc_min_df(collection_parser_config.cooc_min_df());
    config_.set_max_num_of_open_files(500);
    config_.set_num_items_per_batch(collection_parser_config.num_items_per_batch());

    if (!collection_parser_config.has_num_threads() || collection_parser_config.num_threads() < 0) {
      unsigned int n = std::thread::hardware_concurrency();
      if (n == 0) {
        LOG(INFO) << "CollectionParserConfig.num_threads is set to 1 (default)";
        config_.set_num_of_cpu(1);
      } else {
        LOG(INFO) << "CollectionParserConfig.num_threads is automatically set to " << n;
        config_.set_num_of_cpu(n);
      }
    } else {
      config_.set_num_of_cpu(collection_parser_config.num_threads());
    }
  }
}

void CooccurrenceCollector::CreateAndSetTargetFolder() {
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  fs::path dir(boost::lexical_cast<std::string>(uuid));
  if (fs::exists(dir)) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Folder with uuid " +
                                           boost::lexical_cast<std::string>(uuid) +
                                           "already exists"));
  }
  if (!fs::create_directory(dir)) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Failed to create directory"));
  }
  config_.set_target_folder(dir.string());
}

std::string CooccurrenceCollector::CreateFileInBatchDir() const {
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  fs::path file_local_path(boost::lexical_cast<std::string>(uuid));
  fs::path full_filename = fs::path(config_.target_folder()) / file_local_path;
  return full_filename.string();
}

unsigned CooccurrenceCollector::VocabSize() const {
  return vocab_.token_map_.size();
}

void CooccurrenceCollector::ReadVowpalWabbit() {
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
  std::shared_ptr<std::ifstream> vowpal_wabbit_doc_ptr(new std::ifstream(config_.vw_file_path(), std::ios::in));
  if (!vowpal_wabbit_doc_ptr->is_open()) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Failed to open vowpal wabbit file"));
  }
  std::shared_ptr<std::mutex> read_mutex_ptr(new std::mutex);
  std::mutex token_statistics_access;
  std::mutex total_num_of_documents_arg_access;
  std::mutex total_num_of_pairs_arg_access;

  auto func = [&read_mutex_ptr, &token_statistics_access, &total_num_of_documents_arg_access,
               &total_num_of_pairs_arg_access, &vowpal_wabbit_doc_ptr, this]() {
    int64_t local_num_of_pairs = 0;
    while (true) {  // Loop throgh portions.
      // Steps 1-3:
      std::vector<std::string> portion = ReadPortionOfDocuments(read_mutex_ptr, vowpal_wabbit_doc_ptr);
      if (portion.empty()) {
        break;
      }
      {
        std::unique_lock<std::mutex> lock(total_num_of_documents_arg_access);
        total_num_of_documents_ += portion.size();  // statistics for ppmi
      }
      // It will hold tf and df of pairs of tokens
      // Every pair of valid tokens (both exist in vocab) is saved in this storage
      // After walking through portion of documents all the statistics is dumped on disk
      // and then this storage is destroyed
      CooccurrenceStatisticsHolder cooc_stat_holder;

      // For every token from vocab keep the information about the last document this token occured in
      std::vector<unsigned> num_of_last_document_token_occured(vocab_.token_map_.size());

      // When the document is processed (element of portion vector),
      // memory for it can be freed by calling pop_back() from vector
      // (large string will be popped and destroyed)
      // portion.size() can be used as doc_id (it will in method SavePairOfTokens)
      for (; portion.size() != 0; portion.pop_back()) {  // Loop through documents. Step 4:
        std::vector<std::string> doc;
        boost::split(doc, portion.back(), boost::is_any_of(" \t\r"));
        // Step 5.a) Start the loop through a document
        // Loop from 1 because the zero-th element is document title
        std::string first_token_modality = "|@default_class";  // That's how modalities are presented in vw
        for (unsigned j = 1; j < doc.size(); ++j) {  // Loop through tokens in document
          if (doc[j].empty()) {
            continue;
          }
          if (doc[j][0] == '|') {
            first_token_modality = doc[j];
            continue;
          }
          // 5.b) Check if a token is valid
          int first_token_id = vocab_.FindTokenId(doc[j], first_token_modality);
          if (first_token_id == TOKEN_NOT_FOUND) {
            continue;
          }
          // 5.c) Collect statistics for document ppmi (in how many documents every token occured)
          // The array is initialized with zeros, so for every portion.size() it isn't equal to initial value
          if (num_of_last_document_token_occured[first_token_id] != portion.size()) {
            num_of_last_document_token_occured[first_token_id] = portion.size();
            std::unique_lock<std::mutex> lock(token_statistics_access);
            ++num_of_documents_token_occurred_in_[first_token_id];
          }
          // 5.d) Take windows_width tokens (parameter) to the right of the current one
          // If there are some words beginnig on '|' in the text the window should be extended
          // and it's extended using not_a_word_counter
          std::string second_token_modality = first_token_modality;
          unsigned not_a_word_counter = 0;
          // Loop through tokens in the window
          for (unsigned k = 1; k <= config_.cooc_window_width() + not_a_word_counter && j + k < doc.size(); ++k) {
            if (doc[j + k].empty()) {
              continue;
            }
            if (doc[j + k][0] == '|') {
              second_token_modality = doc[j + k];
              ++not_a_word_counter;
              continue;
            }
            if (first_token_modality != second_token_modality) {
              continue;
            }
            int second_token_id = vocab_.FindTokenId(doc[j + k], second_token_modality);
            if (second_token_id == TOKEN_NOT_FOUND) {
              continue;
            }
            // 5.e) When it's known these 2 tokens are valid, remember their co-occurrence
            // Here portion.size() is used to identify a document (it's unique id during one portion of documents)
            if (config_.use_symetric_cooc()) {
              if (first_token_id < second_token_id) {
                cooc_stat_holder.SavePairOfTokens(first_token_id, second_token_id, portion.size());
              } else if (first_token_id > second_token_id) {
                cooc_stat_holder.SavePairOfTokens(second_token_id, first_token_id, portion.size());
              } else {
                cooc_stat_holder.SavePairOfTokens(first_token_id, first_token_id, portion.size(), 2);
              }
            } else {
              cooc_stat_holder.SavePairOfTokens(first_token_id, second_token_id, portion.size());
              cooc_stat_holder.SavePairOfTokens(second_token_id, first_token_id, portion.size());
            }
            local_num_of_pairs += 2;  // statistics for ppmi
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
    }
    {
      std::unique_lock<std::mutex> lock(total_num_of_pairs_arg_access);
      total_num_of_pairs_ += local_num_of_pairs;
    }
  };
  // Launch reading and storing pairs of tokens in parallel
  std::vector<std::shared_future<void>> tasks;
  for (int i = 0; i < config_.num_of_cpu(); ++i) {
    tasks.emplace_back(std::async(std::launch::async, func));
  }
  for (int i = 0; i < config_.num_of_cpu(); ++i) {
    tasks[i].get();
  }
}

std::vector<std::string> CooccurrenceCollector::ReadPortionOfDocuments(
                         std::shared_ptr<std::mutex> read_mutex,
                         std::shared_ptr<std::ifstream> vowpal_wabbit_doc_ptr) {
  std::vector<std::string> portion;
  std::unique_lock<std::mutex> guard(*read_mutex);
  if (vowpal_wabbit_doc_ptr->eof()) {
    return portion;
  }
  std::string str;
  while (static_cast<int>(portion.size()) < config_.num_items_per_batch()) {
    getline(*vowpal_wabbit_doc_ptr, str);
    if (vowpal_wabbit_doc_ptr->eof()) {
      break;
    }
    portion.push_back(std::move(str));
  }
  return portion;
}

void CooccurrenceCollector::UploadOnDisk(const CooccurrenceStatisticsHolder& cooc_stat_holder) {
  // Uploading is implemented as folowing:
  // 1. Create a batch which is associated with a specific file on a disk
  // 2. For every first token id create an object Cell and for every second token
  // that co-occurred with first write it's id, cooc_tf, cooc_df
  // 3. Write the cell in output file and continue the cicle while there are
  // first token ids in cooccurrence statistics holder
  // Note that there can't be two cells stored in ram simultaniously
  // 4. Save batch in vector of objects
  std::shared_ptr<CooccurrenceBatch> batch(CreateNewCooccurrenceBatch());
  OpenBatchOutputFile(batch);
  for (auto iter = cooc_stat_holder.storage_.begin(); iter != cooc_stat_holder.storage_.end(); ++iter) {
    batch->FormNewCell(iter);
    batch->WriteCell();
  }
  CloseBatchOutputFile(batch);
  vector_of_batches_.push_back(std::move(batch));
}

CooccurrenceBatch* CooccurrenceCollector::CreateNewCooccurrenceBatch() const {
  return new CooccurrenceBatch(config_.target_folder());
}

void CooccurrenceCollector::OpenBatchOutputFile(std::shared_ptr<CooccurrenceBatch> batch) {
  if (!batch->out_batch_.is_open()) {
    assert(open_files_counter_ < config_.max_num_of_open_files());
    ++open_files_counter_;
    batch->out_batch_.open(batch->filename_, std::ios::out);
  }
}

void CooccurrenceCollector::CloseBatchOutputFile(std::shared_ptr<CooccurrenceBatch> batch) {
  if (batch->out_batch_.is_open()) {
    --open_files_counter_;
    batch->out_batch_.close();
  }
}

unsigned CooccurrenceCollector::CooccurrenceBatchesQuantity() const {
  return vector_of_batches_.size();
}

void CooccurrenceCollector::ReadAndMergeCooccurrenceBatches() {
  // After that all the statistics has been gathered and saved in form of cooc batches on disk, it
  // needs to be read and merged from cooc batches into one storage
  // If number of cooc batches <= number of files than can be open simultaniously, then
  // all the cooc batches are divided equally into n groups where n = number of cores.
  // Else maximal number of open files is taken and divided into n groups.
  // After that 1 thread has eliberated that can handle another portion of batches than aren't merged yet
  // There are two stages of merging:
  // 1. merging of files of one group (is done asynchroniously, without dropping of rare pairs of tokens)
  // n files are written back in the same form (of cooccurrence batches)
  // If n is too large to merge all the batches into some small number of batches
  // this operation can be in cicle launched many times.
  // 2. then that n files need to be read and merged again (with dropping of rare pairs of tokens)
  // Merging of k files is implemented in KWayMerge function
  // After the second stage the data is written in format of output file (not in format of cooc batches)
  // If there would be a need to calculate ppmi or other values which depend on co-occurrences
  // this data can be read back from output file.

  // std::cout << "Step 2: merging batches" << std::endl;
  const unsigned min_num_of_batches_to_be_merged_in_parallel = 32;
  while (vector_of_batches_.size() > min_num_of_batches_to_be_merged_in_parallel) {
    FirstStageOfMerging();  // size is decreasing here
  }
  ResultingBufferOfCooccurrences res(vocab_, num_of_documents_token_occurred_in_, config_);
  open_files_counter_ += res.open_files_in_buf_;
  SecondStageOfMerging(&res, &vector_of_batches_);
  // std::cout << "Batches have been merged" << std::endl;
  res.CalculatePpmi();
}

void CooccurrenceCollector::FirstStageOfMerging() {
  // Stage 1: merging portions of batches into intermediate batches
  // Note: one thread should merge at least 2 files and have the third to write in
  int tmp = std::min(static_cast<int>(vector_of_batches_.size() / 2), config_.num_of_cpu());
  int num_of_threads = std::min(tmp, config_.max_num_of_open_files() / 3);

  int portion_size = std::min(static_cast<int>(vector_of_batches_.size() / num_of_threads),
                              (config_.max_num_of_open_files() - num_of_threads) / num_of_threads);
  std::shared_ptr<std::mutex> open_close_file_mutex_ptr(new std::mutex);

  auto func = [&open_close_file_mutex_ptr, portion_size, this](int i) {  // Wrapper around KWayMerge
    std::shared_ptr<CooccurrenceBatch> batch(CreateNewCooccurrenceBatch());
    OpenBatchOutputFile(batch);
    ResultingBufferOfCooccurrences intermediate_buffer(vocab_, num_of_documents_token_occurred_in_, config_);
    std::vector<std::shared_ptr<CooccurrenceBatch>> portion_of_batches;
    // Stage 1: take i-th portion from vector_of_batches_
    for (int j = i * portion_size; j < (i + 1) * portion_size &&
                                   j < static_cast<int>(vector_of_batches_.size()); ++j) {
      portion_of_batches.push_back(std::move(vector_of_batches_[j]));
    }
    KWayMerge(&intermediate_buffer, BATCH, &portion_of_batches, batch, open_close_file_mutex_ptr);
    CloseBatchOutputFile(batch);
    return batch;
  };

  std::vector<std::shared_ptr<CooccurrenceBatch>> intermediate_batches;
  std::mutex worker_thread_mutex;
  std::mutex intermediate_batches_access;
  std::queue<unsigned> queue_of_indices;

  auto worker_thread = [&intermediate_batches, &worker_thread_mutex,
                        &intermediate_batches_access, &queue_of_indices, &func]() {
    while (true) {
      unsigned index;
      {  // take one task from queue of tasks
        std::unique_lock<std::mutex> worker_thread_lock(worker_thread_mutex);
        if (queue_of_indices.empty()) {
          break;
        } else {
          index = queue_of_indices.front();
          queue_of_indices.pop();
        }
      }
      std::shared_ptr<CooccurrenceBatch> batch = func(index);
      {
        std::unique_lock<std::mutex> intermediate_batches_access_lock(intermediate_batches_access);
        intermediate_batches.push_back(std::move(batch));
      }
    }
  };
  // Stage 1: prepare indices for threads (each thread will take index and send in func)
  for (int i = 0; i * portion_size < static_cast<int>(vector_of_batches_.size()); ++i) {
    queue_of_indices.push(i);
  }
  // Stage 1: launch workers
  std::vector<std::thread> workers;
  for (int i = 0; i < num_of_threads; ++i) {
    workers.emplace_back(worker_thread);
  }
  for (int i = 0; i < num_of_threads; ++i) {
    workers[i].join();
  }
  vector_of_batches_ = std::move(intermediate_batches);
}

void CooccurrenceCollector::SecondStageOfMerging(ResultingBufferOfCooccurrences* res,
                               std::vector<std::shared_ptr<CooccurrenceBatch>>* intermediate_batches) {
  // Stage 2: merging of final batches (in single thread)
  std::shared_ptr<std::mutex> open_close_file_mutex_ptr(new std::mutex);
  // Note: the 4th arg is fake, it's not used later if mode == OUTPUT_FILE
  KWayMerge(res, OUTPUT_FILE, intermediate_batches, (*intermediate_batches)[0], open_close_file_mutex_ptr);
  // Files are explicitly closed here, because it's necesery to push the data in files on this step
  if (config_.gather_cooc_tf()) {
    res->cooc_tf_dict_out_.close();
  }
  if (config_.gather_cooc_df()) {
    res->cooc_df_dict_out_.close();
  }
  open_files_counter_ -= 2;
}

void CooccurrenceCollector::KWayMerge(ResultingBufferOfCooccurrences* res, const int mode,
                                      std::vector<std::shared_ptr<CooccurrenceBatch>>* vector_of_input_batches_ptr,
                                      std::shared_ptr<CooccurrenceBatch> out_batch,
                                      std::shared_ptr<std::mutex> open_close_file_mutex_ptr) {
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
  // so after merge all the information needed to calculate ppmi will be gathered and
  // available from class ResultingBufferOfCooccurrences
  // Note: here's only 1 way to communicate between threads - through open_files_counter

  // Step 1:
  std::vector<std::shared_ptr<CooccurrenceBatch>>& vector_of_input_batches = *vector_of_input_batches_ptr;
  auto iter = vector_of_input_batches.begin();
  {
    std::unique_lock<std::mutex> open_close_file_lock(*open_close_file_mutex_ptr);
    for (; iter != vector_of_input_batches.end() &&
           open_files_counter_ < config_.max_num_of_open_files() - 1; ++iter) {
      OpenBatchInputFile(*iter);
      (*iter)->ReadCell();
    }
  }
  for (; iter != vector_of_input_batches.end(); ++iter) {
    std::unique_lock<std::mutex> open_close_file_lock(*open_close_file_mutex_ptr);
    OpenBatchInputFile(*iter);
    (*iter)->ReadCell();
    CloseBatchInputFile(*iter);
  }
  // Step 2:
  std::make_heap(vector_of_input_batches.begin(), vector_of_input_batches.end(),
                 CooccurrenceBatch::CoocBatchComparator());
  if (!vector_of_input_batches.empty()) {
    res->cell_ = Cell();
    res->cell_.first_token_id = (*vector_of_input_batches[0]).cell_.first_token_id;
  }
  while (!vector_of_input_batches.empty()) {
    // Step 4:
    if (res->cell_.first_token_id == (*vector_of_input_batches[0]).cell_.first_token_id) {
      res->MergeWithExistingCell(*vector_of_input_batches[0]);
    } else {
      if (mode == BATCH) {
        out_batch->cell_ = res->cell_;
        out_batch->WriteCell();
      } else if (mode == OUTPUT_FILE) {
        if (config_.calculate_ppmi_tf()) {
          res->CalculateTFStatistics();
        }
        if (config_.gather_cooc_tf()) {
          res->WriteCoocFromCell(TokenCoocFrequency, config_.cooc_min_tf());
        }
        if (config_.gather_cooc_df()) {
          res->WriteCoocFromCell(DocumentCoocFrequency, config_.cooc_min_df());
        }
        // It's importants to set size = 0 after popping, because this value will be checked later
        res->cell_.records.resize(0);
      }
      res->cell_ = (*vector_of_input_batches[0]).cell_;
    }
    // Step 3:
    std::pop_heap(vector_of_input_batches.begin(), vector_of_input_batches.end(),
                  CooccurrenceBatch::CoocBatchComparator());
    if (!vector_of_input_batches.back()->in_batch_.is_open()) {
      std::unique_lock<std::mutex> open_close_file_lock(*open_close_file_mutex_ptr);
      OpenBatchInputFile(vector_of_input_batches.back());
    }
    // if there are some data to read ReadCell reads it and returns true, else returns false
    if (vector_of_input_batches.back()->ReadCell()) {
      if (open_files_counter_ == config_.max_num_of_open_files()) {
        std::unique_lock<std::mutex> open_close_file_lock(*open_close_file_mutex_ptr);
        CloseBatchInputFile(vector_of_input_batches.back());
      }
      std::push_heap(vector_of_input_batches.begin(), vector_of_input_batches.end(),
                     CooccurrenceBatch::CoocBatchComparator());
    } else {
      if (IsOpenBatchInputFile(*vector_of_input_batches.back())) {
        std::unique_lock<std::mutex> open_close_file_lock(*open_close_file_mutex_ptr);
        CloseBatchInputFile(vector_of_input_batches.back());
      }
      vector_of_input_batches.pop_back();
    }
  }
  if (!res->cell_.records.empty()) {
    if (mode == BATCH) {
      out_batch->cell_ = res->cell_;
      out_batch->WriteCell();
    } else if (mode == OUTPUT_FILE) {
      if (config_.calculate_ppmi_tf()) {
        res->CalculateTFStatistics();
      }
      if (config_.gather_cooc_tf()) {
        res->WriteCoocFromCell(TokenCoocFrequency, config_.cooc_min_tf());
      }
      if (config_.gather_cooc_df()) {
        res->WriteCoocFromCell(DocumentCoocFrequency, config_.cooc_min_df());
      }
    }
  }
}

void CooccurrenceCollector::OpenBatchInputFile(std::shared_ptr<CooccurrenceBatch> batch) {
  assert(open_files_counter_ < config_.max_num_of_open_files());
  if (!batch->in_batch_.is_open()) {
    ++open_files_counter_;
    batch->in_batch_.open(batch->filename_, std::ios::in);
    batch->in_batch_.seekg(batch->in_batch_offset_);
  }
}

bool CooccurrenceCollector::IsOpenBatchInputFile(const CooccurrenceBatch& batch) const {
  return batch.in_batch_.is_open();
}

void CooccurrenceCollector::CloseBatchInputFile(std::shared_ptr<CooccurrenceBatch> batch) {
  if (batch->in_batch_.is_open()) {
    --open_files_counter_;
    batch->in_batch_offset_ = batch->in_batch_.tellg();
    batch->in_batch_.close();
  }
}

// ********************************** Methods of class Vocab *****************************************

Vocab::Vocab() { }

Vocab::Vocab(const std::string& path_to_vocab) {
  // This function constructs vocab object: reads tokens from vocab file,
  // sets them unique id and collects pairs in unordered map
  std::ifstream vocab_ifile(path_to_vocab, std::ios::in);
  if (!vocab_ifile.good()) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Failed to open vocab file, path = " + path_to_vocab));
  }
  std::string str;
  for (unsigned last_token_id = 0; getline(vocab_ifile, str); ++last_token_id) {
    boost::algorithm::trim(str);
    std::vector<std::string> strs;
    boost::split(strs, str, boost::is_any_of(" \t\r"));
    if (!strs[0].empty()) {
      std::string modality;
      if (strs.size() == 1) {
        modality = "@default_class";  // Here is how modality is indicated in vocab file (without '|')
      } else {
        modality = strs[1];
      }
      std::string key = MakeKey(strs[0], modality);
      auto iter = token_map_.find(key);
      if (iter == token_map_.end()) {
        token_map_.insert(std::make_pair(key, last_token_id));
        inverse_token_map_.insert(std::make_pair(last_token_id, TokenModality(strs[0], modality)));
      } else {
        BOOST_THROW_EXCEPTION(InvalidOperation(
          "There are repeated tokens in vocab file. Please remove all the duplications"));
      }
    }
  }
}

std::string Vocab::MakeKey(const std::string& token_str, const std::string& modality) const {
  return token_str + '|' + modality;
}

int Vocab::FindTokenId(const std::string& token_str, const std::string& modality) const {
  auto token_ref = token_map_.find(MakeKey(token_str, modality));
  if (token_ref == token_map_.end()) {
    return TOKEN_NOT_FOUND;
  }
  return token_ref->second;
}

Vocab::TokenModality Vocab::FindTokenStr(const int token_id) const {
  auto token_ref = inverse_token_map_.find(token_id);
  if (token_ref == inverse_token_map_.end()) {
    return TokenModality();
  }
  return token_ref->second;
}

// ****************************** Methods of class CooccurrenceStatisticsHolder *************************************

// This class stores temporarily added statistics about pairs of tokens (how often these pairs occurred in documents
// in a window and in how many documents they occurred together in a window).
// The data is stored in rb tree (std::map)
void CooccurrenceStatisticsHolder::SavePairOfTokens(const int first_token_id, const int second_token_id,
                                                    const unsigned doc_id, const double weight) {
  // There are 2 levels of indexing
  // The first level keeps information about first token and the second level
  // about co-occurrence between the first and the second tokens
  // If first token id is known (exists in the structure), corresponding node should be modified
  // else it should be added to the structure
  auto it1 = storage_.find(first_token_id);
  if (it1 == storage_.end()) {
    FirstToken first_token;
    first_token.second_token_reference = std::map<int, SecondTokenAndCooccurrence> {
      {second_token_id, SecondTokenAndCooccurrence(doc_id, weight)}
    };
    storage_.insert(std::pair<int, FirstToken>(first_token_id, first_token));
  } else {
    std::map<int, SecondTokenAndCooccurrence>& second_tokens = it1->second.second_token_reference;
    auto it2 = second_tokens.find(second_token_id);
    if (it2 == second_tokens.end()) {
      SecondTokenAndCooccurrence second_token(doc_id, weight);
      second_tokens.insert(std::pair<int, SecondTokenAndCooccurrence>(second_token_id, second_token));
    } else {
      if (it2->second.last_doc_id != doc_id) {
        it2->second.last_doc_id = doc_id;
        ++(it2->second.cooc_df);
      }
      it2->second.cooc_tf += weight;
    }
  }
}

bool CooccurrenceStatisticsHolder::Empty() {
  return storage_.empty();
}

// ************************************** Methods of class CooccurrenceBatch ******************************************

CooccurrenceBatch::CooccurrenceBatch(const std::string& path_to_batches) : in_batch_offset_(0) {
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  fs::path batch(boost::lexical_cast<std::string>(uuid));
  fs::path full_filename = fs::path(path_to_batches) / batch;
  filename_ = full_filename.string();
}

void CooccurrenceBatch::FormNewCell(const std::map<int, CooccurrenceStatisticsHolder::FirstToken>
                                             ::const_iterator& cooc_stat_node) {
  // Here is initialization of a new cell
  // A cell consists on first_token_id, number of records it includes
  // Then records go, every reord consists on second_token_id, cooc_tf, cooc_df

  cell_.first_token_id = cooc_stat_node->first;
  // while reading from file it's necessery to know how many records to read
  auto second_token_reference = cooc_stat_node->second.second_token_reference;
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
    BOOST_THROW_EXCEPTION(InvalidOperation("Error while reading from batch. File is corrupted"));
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
    const Vocab& vocab,
    const std::vector<unsigned>& num_of_documents_token_occurred_in,
    const CooccurrenceCollectorConfig& config) : vocab_(vocab),
                      num_of_documents_token_occurred_in_(num_of_documents_token_occurred_in),
                      config_(config) {
  num_of_pairs_token_occurred_in_.resize(vocab_.token_map_.size());
  // ToDo (MichaelSolotky): make it easier to read
  if (config_.has_gather_cooc_tf()) {  // It's important firstly to create file (open output file)
    cooc_tf_dict_out_.open(config_.cooc_tf_file_path(), std::ios::out);
    CheckOutputFile(cooc_tf_dict_out_, config_.cooc_tf_file_path());
    cooc_tf_dict_in_.open(config_.cooc_tf_file_path(), std::ios::in);
    CheckInputFile(cooc_tf_dict_in_, config_.cooc_tf_file_path());
  }
  if (config_.has_gather_cooc_df()) {
    cooc_df_dict_out_.open(config_.cooc_df_file_path(), std::ios::out);
    CheckOutputFile(cooc_df_dict_out_, config_.cooc_df_file_path());
    cooc_df_dict_in_.open(config_.cooc_df_file_path(), std::ios::in);
    CheckInputFile(cooc_df_dict_in_, config_.cooc_df_file_path());
  }
  if (config_.has_calculate_ppmi_tf()) {
    ppmi_tf_dict_.open(config_.ppmi_tf_file_path(), std::ios::out);
    CheckOutputFile(ppmi_tf_dict_, config_.ppmi_tf_file_path());
  }
  if (config_.has_calculate_ppmi_df()) {
    ppmi_df_dict_.open(config_.ppmi_df_file_path(), std::ios::out);
    CheckOutputFile(ppmi_df_dict_, config_.ppmi_df_file_path());
  }
}

void ResultingBufferOfCooccurrences::CheckInputFile(const std::ifstream& file, const std::string& filename) {
  if (!file.good()) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Failed to open input file " +
                                            filename + " in working directory"));
  }
}

void ResultingBufferOfCooccurrences::CheckOutputFile(const std::ofstream& file, const std::string& filename) {
  if (!file.good()) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Failed to open or create output file " +
                                            filename + " in working directory"));
  }
}

// ToDo (MichaelSolotky): may be this can be implemented in more optimal way
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
    } else if (fi_iter->second_token_id < se_iter->second_token_id) {
      *th_iter = *fi_iter;
      ++fi_iter;
    } else {
      *th_iter = *se_iter;
      ++se_iter;
    }
    ++th_iter;
  }
  cell_.records.resize(th_iter - cell_.records.begin());
  std::copy(fi_iter, old_vector.end(), std::back_inserter(cell_.records));
  std::copy(se_iter, batch.cell_.records.end(), std::back_inserter(cell_.records));
}

void ResultingBufferOfCooccurrences::CalculateTFStatistics() {
  // Calculate statistics of occurrence (of first token which is associated with current cell)
  int64_t n_u = 0;
  for (unsigned i = 0; i < cell_.records.size(); ++i) {
    if (config_.use_symetric_cooc() && cell_.first_token_id != cell_.records[i].second_token_id) {
      num_of_pairs_token_occurred_in_[cell_.records[i].second_token_id] += cell_.records[i].cooc_tf;
    }  // pairs <u u> have double weight so in symetric case they should be taken once
    n_u += cell_.records[i].cooc_tf;
  }
  num_of_pairs_token_occurred_in_[cell_.first_token_id] += n_u;
}

void ResultingBufferOfCooccurrences::WriteCoocFromCell(const std::string mode, const unsigned cooc_min) {
  // This function takes a cell from buffer and writes data from cell in file
  // Output file format(s) are defined here
  // stringstream is used for fast bufferized i/o operations
  // Note: before writing in file all the information is stored in ram
  std::stringstream output_buf;
  bool no_cooc_found = true;
  std::string prev_modality = "@default_class";
  Vocab::TokenModality first_token = vocab_.FindTokenStr(cell_.first_token_id);
  if (first_token.modality != "@default_class") {
    output_buf << '|' << first_token.modality << ' ';
    prev_modality = first_token.modality;
  }
  output_buf << first_token.token_str << ' ';
  for (unsigned i = 0; i < cell_.records.size(); ++i) {
    if (cell_.GetCoocFromCell(mode, i) >= cooc_min && cell_.first_token_id != cell_.records[i].second_token_id) {
      no_cooc_found = false;
      Vocab::TokenModality second_token = vocab_.FindTokenStr(cell_.records[i].second_token_id);
      if (second_token.modality != prev_modality) {
        output_buf << " |" << second_token.modality << ' ';
        prev_modality = second_token.modality;
      }
      output_buf << second_token.token_str << ':' << cell_.GetCoocFromCell(mode, i) << ' ';
    }
  }
  if (!no_cooc_found) {
    output_buf << '\n';
    if (mode == TokenCoocFrequency) {
      cooc_tf_dict_out_ << output_buf.str();
    } else {
      cooc_df_dict_out_ << output_buf.str();
    }
  }
}

void ResultingBufferOfCooccurrences::CalculatePpmi() {  // Wrapper around CalculateAndWritePpmi
  // std::cout << "Step 3: start calculation ppmi" << std::endl;
  if (config_.calculate_ppmi_tf()) {
    CalculateAndWritePpmi(TokenCoocFrequency, config_.total_num_of_pairs());
  }
  if (config_.calculate_ppmi_df()) {
    CalculateAndWritePpmi(DocumentCoocFrequency, config_.total_num_of_documents());
  }
  // std::cout << "Ppmi's have been calculated" << std::endl;
}

void ResultingBufferOfCooccurrences::CalculateAndWritePpmi(const std::string mode, const long double n) {
  // This function reads cooc file line by line, calculates ppmi and saves them in external file of ppmi
  // stringstream is used for fast bufferized i/o operations
  // Note: before writing in file all the information is stored in ram
  std::stringstream output_buf;
  std::string str;
  while (getline(mode == TokenCoocFrequency ? cooc_tf_dict_in_ : cooc_df_dict_in_, str)) {
    boost::algorithm::trim(str);
    std::string first_token_modality = DefaultClass;  // Here's how modality is indicated in output file
    bool new_first_token = true;
    std::vector<std::string> strs;
    boost::split(strs, str, boost::is_any_of(" :\t\r"));
    unsigned index_of_first_token = 0;
    // Find modality
    for (; index_of_first_token < strs.size() && (strs[index_of_first_token].empty() ||
                                                  strs[index_of_first_token][0] == '|'); ++index_of_first_token) {
      if (!strs[index_of_first_token].empty()) {
        first_token_modality = strs[index_of_first_token].substr(1);
      }
    }
    std::string first_token_str = strs[index_of_first_token];
    unsigned not_a_word_counter = 0;
    std::string prev_modality = first_token_modality;
    for (unsigned i = index_of_first_token + 1; i + not_a_word_counter < strs.size(); i += 2) {
      std::string second_token_modality = first_token_modality;
      for (; i + not_a_word_counter < strs.size() && (strs[i + not_a_word_counter].empty() ||
                                                      strs[i + not_a_word_counter][0] == '|'); ++not_a_word_counter) {
        if (!strs[i + not_a_word_counter].empty()) {
          second_token_modality = strs[i + not_a_word_counter].substr(1);
        }
      }
      if (i + not_a_word_counter + 1 >= strs.size()) {
        break;
      }
      std::string second_token_str = strs[i + not_a_word_counter];
      int64_t cooc = std::stoull(strs[i + not_a_word_counter + 1]);
      long double n_u = GetTokenFreq(mode, vocab_.FindTokenId(first_token_str, first_token_modality));
      long double n_v = GetTokenFreq(mode, vocab_.FindTokenId(second_token_str, second_token_modality));
      long double n_uv = static_cast<long double>(cooc);
      double value_inside_logarithm = (n / n_u) / (n_v / n_uv);
      if (value_inside_logarithm > 1.0) {
        if (new_first_token) {
          if (first_token_modality != DefaultClass) {
            output_buf << first_token_modality << ' ';
          }
          output_buf << first_token_str;
          new_first_token = false;
        }
        if (second_token_modality != prev_modality) {
          output_buf << ' ' << second_token_modality;
          prev_modality = second_token_modality;
        }
        output_buf << ' ' << second_token_str << ':' << log(value_inside_logarithm);
      }
    }
    if (!new_first_token) {
      output_buf << '\n';
    }
  }
  if (mode == TokenCoocFrequency) {
    ppmi_tf_dict_ << output_buf.str();
  } else {
    ppmi_df_dict_ << output_buf.str();
  }
}

double ResultingBufferOfCooccurrences::GetTokenFreq(const std::string& mode, const int token_id) const {
  if (mode == TokenCoocFrequency) {
    return num_of_pairs_token_occurred_in_[token_id];
  } else {
    return num_of_documents_token_occurred_in_[token_id];
  }
}

class FileWrapper {  // ToDo (MichaelSolotky): finish it (and replace fsrteams with c-style files)
 public:
  FileWrapper(const std::string& filename, const std::string mode) :
              file_ptr_(fopen(filename.c_str(), mode.c_str())) { }

 private:
  FILE* file_ptr_;
};

}  // namespace core
}  // namespace artm
// vim: set ts=2 sw=2:
