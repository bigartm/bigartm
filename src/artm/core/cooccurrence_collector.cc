// Copyright 2019, Additive Regularization of Topic Models.

#include "artm/core/cooccurrence_collector.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <future>  // NOLINT
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/utility.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

#include "artm/core/collection_parser.h"
#include "artm/core/common.h"
#include "artm/core/exceptions.h"
#include "artm/core/token.h"

namespace fs = boost::filesystem;

namespace artm {
namespace core {

// ToDo (MichaelSolotky): add this input format to batches, (optionally) Dictionary
// 1. find a point in code after collection parsing where this code can be inserted
// 2. take parser config from Parse method and find batch folder name
// 3. use method ListAllBatches to have vector of filenames
// 4. read batches in some order and count co-occurrences (use method LoadBatches)
// 4.a understand how to extract text from batches
// 5. think how to specify ppmi and cooc output files
// 6. remove mutexes somewhere and make the code faster (even with little mistakes in clalculations)

// ToDo (MichaelSolotky): search for all bad-written parts of code with CLion

// ************************************ Methods of class CooccurrenceCollector ************************************

CooccurrenceCollector::CooccurrenceCollector(
      const CollectionParserConfig& collection_parser_config) : open_files_counter_(0) {
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
    config_.set_num_items_per_batch(collection_parser_config.num_items_per_batch());

    // This is the maximal allowable number. With larger values there are problems in Mac OS
    // (experimentally found)
    int max_num_of_open_files_in_a_process = 251;
    // The subtraction was done in order to run on all the machines
    // Maybe in the other systems this limit is lower and without subtracion this code would crash
    // (although it works on my MacOS Mojave 10.14.2)
    max_num_of_open_files_in_a_process -= 10;
    config_.set_max_num_of_open_files_in_a_process(max_num_of_open_files_in_a_process);

    if (!collection_parser_config.has_num_threads() || collection_parser_config.num_threads() < 0) {
      int n = std::thread::hardware_concurrency();
      if (n == 0) {
        config_.set_num_threads(1);
        LOG(INFO) << "CooccurrenceCollectorConfig.num_threads is set to 1 (default)";
      } else {
        // In agglomerative merge a larger number of threads would be suboptimal
        config_.set_num_threads(std::min(n, max_num_of_open_files_in_a_process / 3));
        LOG(INFO) << "CooccurrenceCollectorConfig.num_threads is automatically set to " << config_.num_threads();
      }
    } else {
      // In agglomerative merge a larger number of threads would be suboptimal
      config_.set_num_threads(std::min(collection_parser_config.num_threads(),
                                       max_num_of_open_files_in_a_process / 3));
    }

    const int max_num_of_open_files_in_a_thread = max_num_of_open_files_in_a_process / config_.num_threads();
    config_.set_max_num_of_open_files_in_a_thread(max_num_of_open_files_in_a_thread);
  }
}

std::string CooccurrenceCollector::MakeKeyForVocab(const std::string& token_str, const std::string& modality) const {
  return vocab_.MakeKey(token_str, modality);
}

int CooccurrenceCollector::FindTokenIdInVocab(const std::string& token_str, const std::string& modality) {
  std::unique_lock<std::mutex> vocab_access_lock(vocab_access_mutex_);
  return vocab_.FindTokenId(token_str, modality);
}

Vocab::TokenModality CooccurrenceCollector::FindTokenStrInVocab(const int token_id) {
  std::unique_lock<std::mutex> vocab_access_lock(vocab_access_mutex_);
  return vocab_.FindTokenStr(token_id);
}

unsigned CooccurrenceCollector::VocabSize() {
  std::unique_lock<std::mutex> vocab_access_lock(vocab_access_mutex_);
  return vocab_.VocabSize();
}

// Note: the method is no longer used
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
  {
    std::unique_lock<std::mutex> vector_of_batches_access_lock(vector_of_batches_access_mutex_);
    vector_of_batches_.push_back(std::move(batch));
  }
}

CooccurrenceBatch* CooccurrenceCollector::CreateNewCooccurrenceBatch() {
  std::unique_lock<std::mutex> target_dir_access_lock(target_dir_access_mutex_);
  return new CooccurrenceBatch(config_.target_folder());
}

void CooccurrenceCollector::OpenBatchOutputFile(std::shared_ptr<CooccurrenceBatch> batch) {
  if (!batch->out_batch_.is_open()) {
    std::unique_lock<std::mutex> open_close_file_lock(open_close_file_mutex_);
    assert(open_files_counter_ < config_.max_num_of_open_files_in_a_process());
    batch->out_batch_.open(batch->filename_, std::ios::out);
    if (!batch->out_batch_.is_open()) {
      BOOST_THROW_EXCEPTION(InvalidOperation(
        "Failed to open co-occurrence batch file for writing, path = " + batch->filename_));
    }
    ++open_files_counter_;
  }
}

void CooccurrenceCollector::CloseBatchOutputFile(std::shared_ptr<CooccurrenceBatch> batch) {
  if (batch->out_batch_.is_open()) {
    std::unique_lock<std::mutex> open_close_file_lock(open_close_file_mutex_);
    batch->out_batch_.close();
    if (batch->out_batch_.is_open()) {
      BOOST_THROW_EXCEPTION(InvalidOperation(
        "Failed to close co-occurrence batch file, path = " + batch->filename_));
    }
    --open_files_counter_;
  }
}

unsigned CooccurrenceCollector::NumOfCooccurrenceBatches() const {
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

  std::cerr << "\nMerging co-occurrence batches. Stage 1: parallel agglomerative merge" << std::endl;
  // This magic constant isn't the best and can be set to another value found in experiments
  // This value should be lower than maximal number of open files in a process to avoid frequent
  // openning-closing of the files
  const unsigned min_num_of_batches_to_be_merged_in_parallel = 32;
  while (NumOfCooccurrenceBatches() > min_num_of_batches_to_be_merged_in_parallel) {
    FirstStageOfMerging();  // number of files is decreasing here
  }
  std::cerr << "Merging co-occurrence batches. Stage 2: sequential merge" << std::endl;
  // vocab will be coppyied here into buffer_for_output_files
  BufferOfCooccurrences buffer_for_output_files(OUTPUT_FILE, vocab_, num_of_documents_token_occurred_in_, config_);
  open_files_counter_ += buffer_for_output_files.open_files_counter_;
  SecondStageOfMerging(&buffer_for_output_files, &vector_of_batches_);
  buffer_for_output_files.CalculatePpmi();
  open_files_counter_ -= buffer_for_output_files.open_files_counter_;
}

// ToDo (MichaelSolotky): remove batches after merging
void CooccurrenceCollector::FirstStageOfMerging() {
  // Stage 1: merging portions of batches into intermediate batches
  // The strategy is the folowing: divide the vector of batches into as much buckets as it's possible
  // and gives advantages (the best case is 2 batches into each bucket, and all these batches are open
  // simultaneously), then create several threads which will take a portion of batches, merge them,
  // then a free thread will take another not-merged portion of batches and so on until
  // there are no portions left. This procedure produces new batches, but their number is much lower
  // then but their number is much smaller than those which were sent at the input,
  // so it could be needed to run it again in order to make that number small enough
  const int num_of_batches = static_cast<int>(vector_of_batches_.size());
  // Let k be number of buckets in which all the batches will be divided
  // Let n be total number of batches
  // k ≤ upper bound of n / 2
  const int upper_constraint_due_to_num_of_batches = num_of_batches / 2 + num_of_batches % 2;
  // The procedure would be uneffective if there had been some buckets with at least 1 closed file,
  // so there should be at least 3 open files in each bucket: 1 for writing and at least 2 merging files
  const int upper_constraint_due_to_num_of_open_files = config_.max_num_of_open_files_in_a_thread() / 3;
  const int optimal_num_of_threads = std::min(upper_constraint_due_to_num_of_batches,
                                              upper_constraint_due_to_num_of_open_files);
  int num_of_threads = optimal_num_of_threads;
  // The real number of threads shouldn't be higher the than value set by the user of number of cores in computer
  if (config_.has_num_threads() && config_.num_threads() > 0) {
    num_of_threads = std::min(optimal_num_of_threads, config_.num_threads());
  }
  // 1 is subtracted here because 1 file will be needed for writing in it
  const int optimal_portion_size = std::min(static_cast<int>(vector_of_batches_.size() / num_of_threads),
                                            config_.max_num_of_open_files_in_a_thread() - 1);
  // Portion size < 2 would lead to cycling or an error,
  // if it would be > 10, much RAM space would be needed and i/o operations would be to frequent that
  // the merging would slow due to pauses for reading and writing
  const int portion_size = std::min(std::max(optimal_portion_size, 2), 10);
  std::shared_ptr<std::mutex> open_close_file_mutex_ptr(new std::mutex);

  auto func = [&open_close_file_mutex_ptr, portion_size, this](int portion_index) {  // Wrapper around KWayMerge
    const int total_num_of_batches = vector_of_batches_.size();
    std::shared_ptr<CooccurrenceBatch> batch(CreateNewCooccurrenceBatch());
    OpenBatchOutputFile(batch);
    // vocab will be coppied here into buffer_for_a_batch
    BufferOfCooccurrences buffer_for_a_batch(BATCH, vocab_,
                                             num_of_documents_token_occurred_in_,
                                             config_);
    std::vector<std::shared_ptr<CooccurrenceBatch>> portion_of_batches;
    // Stage 1: take i-th portion from pull of batches
    {
      std::unique_lock<std::mutex> vector_of_batches_access_lock(vector_of_batches_access_mutex_);
      for (int batch_index = portion_index * portion_size;
           batch_index < (portion_index + 1) * portion_size && batch_index < total_num_of_batches;
           ++batch_index) {
        portion_of_batches.push_back(std::move(vector_of_batches_[batch_index]));
      }
    }
    KWayMerge(&buffer_for_a_batch, BATCH, &portion_of_batches, batch, open_close_file_mutex_ptr);
    CloseBatchOutputFile(batch);
    return batch;
  };

  std::vector<std::shared_ptr<CooccurrenceBatch>> intermediate_batches;
  std::mutex intermediate_batches_access_mutex;
  std::queue<unsigned> queue_of_indices;
  std::mutex queue_access_mutex;

  auto worker = [&intermediate_batches, &intermediate_batches_access_mutex,
                 &queue_of_indices, &queue_access_mutex, &func]() {
    while (true) {
      unsigned index;
      {  // take one task from queue of tasks
        std::unique_lock<std::mutex> queue_access_lock(queue_access_mutex);
        if (queue_of_indices.empty()) {
          break;
        } else {
          index = queue_of_indices.front();
          queue_of_indices.pop();
        }
      }
      std::shared_ptr<CooccurrenceBatch> batch = func(index);
      {
        std::unique_lock<std::mutex> intermediate_batches_access_lock(intermediate_batches_access_mutex);
        intermediate_batches.push_back(std::move(batch));
      }
    }
  };
  // Prepare indices for threads (each thread will take index and send in func)
  for (int i = 0; i * portion_size < static_cast<int>(vector_of_batches_.size()); ++i) {
    queue_of_indices.push(i);
  }
  // Launch workers
  std::vector<std::thread> workers;
  for (int i = 0; i < num_of_threads; ++i) {
    workers.emplace_back(worker);
  }
  for (int i = 0; i < num_of_threads; ++i) {
    workers[i].join();
  }
  vector_of_batches_ = std::move(intermediate_batches);
}

void CooccurrenceCollector::SecondStageOfMerging(BufferOfCooccurrences* buffer_for_output_files,
                               std::vector<std::shared_ptr<CooccurrenceBatch>>* intermediate_batches) {
  // Stage 2: merging of final batches (in single thread)
  std::shared_ptr<std::mutex> open_close_file_mutex_ptr(new std::mutex);
  // Note: the 4th arg is fake, it's not used later if mode == OUTPUT_FILE
  KWayMerge(buffer_for_output_files, OUTPUT_FILE, intermediate_batches,
            (*intermediate_batches)[0], open_close_file_mutex_ptr);
  // Files are explicitly closed here, because it's necesery to push the data in files on this step
  if (config_.gather_cooc_tf()) {
    buffer_for_output_files->cooc_tf_dict_out_.close();
  }
  if (config_.gather_cooc_df()) {
    buffer_for_output_files->cooc_df_dict_out_.close();
  }
  buffer_for_output_files->open_files_counter_ -= 2;
  open_files_counter_ -= 2;
}

void CooccurrenceCollector::KWayMerge(BufferOfCooccurrences* buffer, const int mode,
                                      std::vector<std::shared_ptr<CooccurrenceBatch>>* vector_of_input_batches_ptr,
                                      std::shared_ptr<CooccurrenceBatch> out_batch,
                                      std::shared_ptr<std::mutex> open_close_file_mutex_ptr) {
  // All cooc batches has it's local buffer in RAM (look the CooccurrenceBatch class implementation)
  // Information in batches is stored in cells.
  // There are 2 different output formats, which are set via mode parameter:
  // 1. Batches
  // 2. Output file
  // Here's the k-way merge algorithm for external sorting:
  // 1. Initially first cells of all the batches are read in their buffers
  // 2. Then batches are sorted (std::make_heap) by first_token_id of the cell
  // 3. Then a cell with the lowest first_token_id is extaracted and put in
  // buffer and the next cell is read from corresponding batch
  // 4. If the lowest first token id equals first token id of cell of this buffer, they are merged
  // else the current cell is written in file and the new one is loaded.
  // Writing and empting is done in order to keep low memory consumption
  // During execution of this function (if mode is OUTPUT_FILE) n_u is calculated and saved,
  // so after merge all the information needed to calculate ppmi will be gathered and
  // available from class BufferOfCooccurrences
  // Note: here's only 1 way to communicate between threads - through open_files_counter

  // Step 1:
  std::vector<std::shared_ptr<CooccurrenceBatch>>& vector_of_input_batches = *vector_of_input_batches_ptr;
  auto iter = vector_of_input_batches.begin();
  {  // Open all files and read the first cell
    std::unique_lock<std::mutex> open_close_file_lock(*open_close_file_mutex_ptr);
    for (; iter != vector_of_input_batches.end(); ++iter) {
      OpenBatchInputFile(*iter);
      (*iter)->ReadCell();
    }
  }
  // Step 2:
  std::make_heap(vector_of_input_batches.begin(), vector_of_input_batches.end(),
                 CooccurrenceBatch::CoocBatchComparator());
  if (!vector_of_input_batches.empty()) {
    buffer->cell_ = Cell();
    buffer->cell_.first_token_id = (*vector_of_input_batches[0]).cell_.first_token_id;
  }
  while (!vector_of_input_batches.empty()) {
    // Step 4:
    if (buffer->cell_.first_token_id == (*vector_of_input_batches[0]).cell_.first_token_id) {
      buffer->MergeWithExistingCell(*vector_of_input_batches[0]);
    } else {
      if (mode == BATCH) {
        out_batch->cell_ = buffer->cell_;
        out_batch->WriteCell();
      } else if (mode == OUTPUT_FILE) {
        if (config_.calculate_ppmi_tf()) {
          buffer->CalculateTFStatistics();
        }
        if (config_.gather_cooc_tf()) {
          buffer->WriteCoocFromCell(TokenCoocFrequency, config_.cooc_min_tf());
        }
        if (config_.gather_cooc_df()) {
          buffer->WriteCoocFromCell(DocumentCoocFrequency, config_.cooc_min_df());
        }
        // It's importants to set size = 0 after popping, because this value will be checked later
        buffer->cell_.records.resize(0);
      }
      buffer->cell_ = (*vector_of_input_batches[0]).cell_;
    }
    // Step 3:
    std::pop_heap(vector_of_input_batches.begin(), vector_of_input_batches.end(),
                  CooccurrenceBatch::CoocBatchComparator());
    // if there are some data to read ReadCell reads it and returns true, else returns false
    if (vector_of_input_batches.back()->ReadCell()) {
      std::push_heap(vector_of_input_batches.begin(), vector_of_input_batches.end(),
                     CooccurrenceBatch::CoocBatchComparator());
    } else {
      std::unique_lock<std::mutex> open_close_file_lock(*open_close_file_mutex_ptr);
      CloseBatchInputFile(vector_of_input_batches.back());
      vector_of_input_batches.pop_back();
    }
  }
  if (!buffer->cell_.records.empty()) {
    if (mode == BATCH) {
      out_batch->cell_ = buffer->cell_;
      out_batch->WriteCell();
    } else if (mode == OUTPUT_FILE) {
      if (config_.calculate_ppmi_tf()) {
        buffer->CalculateTFStatistics();
      }
      if (config_.gather_cooc_tf()) {
        buffer->WriteCoocFromCell(TokenCoocFrequency, config_.cooc_min_tf());
      }
      if (config_.gather_cooc_df()) {
        buffer->WriteCoocFromCell(DocumentCoocFrequency, config_.cooc_min_df());
      }
    }
  }
}

void CooccurrenceCollector::OpenBatchInputFile(std::shared_ptr<CooccurrenceBatch> batch) {
  if (!batch->in_batch_.is_open()) {
    std::unique_lock<std::mutex> open_close_file_lock(open_close_file_mutex_);
    assert(open_files_counter_ < config_.max_num_of_open_files_in_a_process());
    batch->in_batch_.open(batch->filename_, std::ios::in);
    if (!batch->in_batch_.is_open()) {
      BOOST_THROW_EXCEPTION(InvalidOperation(
        "Failed to open co-occurrence batch file for reading, path = " + batch->filename_));
    }
    // ToDo (MichaelSolotky): find the way to check correctness of the seekg procedure
    batch->in_batch_.seekg(batch->in_batch_offset_);
    ++open_files_counter_;
  }
}

bool CooccurrenceCollector::IsOpenBatchInputFile(const CooccurrenceBatch& batch) const {
  return batch.in_batch_.is_open();
}

void CooccurrenceCollector::CloseBatchInputFile(std::shared_ptr<CooccurrenceBatch> batch) {
  if (batch->in_batch_.is_open()) {
    std::unique_lock<std::mutex> open_close_file_lock(open_close_file_mutex_);
    batch->in_batch_offset_ = batch->in_batch_.tellg();
    batch->in_batch_.close();
    if (batch->in_batch_.is_open()) {
      BOOST_THROW_EXCEPTION(InvalidOperation(
        "Failed to close co-occurrence batch file, path = " + batch->filename_));
    }
    --open_files_counter_;
  }
}

// ********************************** Methods of class Vocab **********************************

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
    boost::split(strs, str, boost::is_any_of(" ,\t\r"));
    if (!strs[0].empty()) {
      std::string modality;
      int pos_of_modality = 1;
      for (; pos_of_modality < strs.size() && strs[pos_of_modality].empty(); ++pos_of_modality) { }
      if (pos_of_modality != strs.size()) {
        modality = strs[pos_of_modality];
      } else {
        modality = DefaultClass;
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

unsigned Vocab::VocabSize() const {
  return token_map_.size();
}

// ****************************** Methods of class CooccurrenceStatisticsHolder ******************************

// This class stores temporarily added statistics about pairs of tokens (how often these pairs
// occurred in documents in a window and in how many documents they occurred together in a window).
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

// ******************************** Methods of class CooccurrenceBatch ********************************

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

// ********************************* Methods of class BufferOfCooccurrences *********************************

// ToDo (MichaeSolotky): logic of this class isn't obvious, it would be cool to express thoughts clearer
// ToDo (MichaeSolotky): don't copy vocab if there's no need (like in CooccurrenceCollector)

// The main purpose of this class is to store statistics of co-occurrences and some
// variables calculated on base of them, perform that calculations, write that in
// target file and read from it.
// This class stores data in cells (special structure) before they are written in resulting files
// A cell can from a batch can come in this buffer and be merged with current stored
// (in case first_token_ids of cells are equal) or the current cell can be pushed from buffer in file
// (in case first_token_ids aren't equal) and new cell takes place of the old one
// This buffer can be used for writing in a batch or in a target file (like file of co-occurrences
// or file of pPMI), the target_ parameter is responsible for this, it can be either
// OUTPUT_FILE or BATCH
BufferOfCooccurrences::BufferOfCooccurrences(
    const int target, const Vocab& vocab,
    const std::vector<unsigned>& num_of_documents_token_occurred_in,
    const CooccurrenceCollectorConfig& config) : target_(target), vocab_(vocab),
                      num_of_documents_token_occurred_in_(num_of_documents_token_occurred_in),
                      open_files_counter_(0), config_(config) {
  num_of_pairs_token_occurred_in_.resize(vocab_.token_map_.size());
  if (target_ == OUTPUT_FILE) {  // Open that files only if planning to write in them
    if (config_.gather_cooc_tf()) {  // It's important firstly to create file (open output file)
      cooc_tf_dict_out_.open(config_.cooc_tf_file_path(), std::ios::out);
      CheckOutputFile(cooc_tf_dict_out_, config_.cooc_tf_file_path());
      cooc_tf_dict_in_.open(config_.cooc_tf_file_path(), std::ios::in);
      CheckInputFile(cooc_tf_dict_in_, config_.cooc_tf_file_path());
      open_files_counter_ += 2;
    }
    if (config_.gather_cooc_df()) {
      cooc_df_dict_out_.open(config_.cooc_df_file_path(), std::ios::out);
      CheckOutputFile(cooc_df_dict_out_, config_.cooc_df_file_path());
      cooc_df_dict_in_.open(config_.cooc_df_file_path(), std::ios::in);
      CheckInputFile(cooc_df_dict_in_, config_.cooc_df_file_path());
      open_files_counter_ += 2;
    }
    if (config_.calculate_ppmi_tf()) {
      ppmi_tf_dict_.open(config_.ppmi_tf_file_path(), std::ios::out);
      CheckOutputFile(ppmi_tf_dict_, config_.ppmi_tf_file_path());
      ++open_files_counter_;
    }
    if (config_.calculate_ppmi_df()) {
      ppmi_df_dict_.open(config_.ppmi_df_file_path(), std::ios::out);
      CheckOutputFile(ppmi_df_dict_, config_.ppmi_df_file_path());
      ++open_files_counter_;
    }
  }
}

void BufferOfCooccurrences::CheckInputFile(const std::ifstream& file, const std::string& filename) {
  if (!file.good()) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Failed to open input file " +
                                            filename + " in working directory"));
  }
}

void BufferOfCooccurrences::CheckOutputFile(const std::ofstream& file, const std::string& filename) {
  if (!file.good()) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Failed to open or create output file " +
                                            filename + " in working directory"));
  }
}

// ToDo (MichaelSolotky): may be this can be implemented in more optimal way
void BufferOfCooccurrences::MergeWithExistingCell(const CooccurrenceBatch& batch) {
  // All the data in buffer are stored in a cell, so here are rules of updating each cell
  // This function takes two vectors (one of the current cell and one which is stored in batch),
  // merges them in folowing way:
  // 1. If two elements of vector are different (different second token id),
  // stacks them one to another in ascending order
  // 2. It these two elemnts are equal, adds their cooc_tf and cooc_df and
  // stores fianl cell with this parameters
  // After merging final vector is sorted in ascending order
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

void BufferOfCooccurrences::CalculateTFStatistics() {
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

void BufferOfCooccurrences::WriteCoocFromCell(const std::string mode, const unsigned cooc_min) {
  // This function takes a cell from buffer and writes data from cell in file
  // Output file format(s) are defined here
  // stringstream is used for fast bufferized i/o operations
  // Note: before writing in file all the information is stored in ram
  std::stringstream output_buf;
  bool no_cooc_found = true;
  std::string prev_modality = DefaultClass;
  Vocab::TokenModality first_token = vocab_.FindTokenStr(cell_.first_token_id);
  if (first_token.modality != DefaultClass) {
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

void BufferOfCooccurrences::CalculatePpmi() {  // Wrapper around CalculateAndWritePpmi
  std::cerr << "Calculating pPMI" << std::endl;
  if (config_.calculate_ppmi_tf()) {
    CalculateAndWritePpmi(TokenCoocFrequency, config_.total_num_of_pairs());
  }
  if (config_.calculate_ppmi_df()) {
    CalculateAndWritePpmi(DocumentCoocFrequency, config_.total_num_of_documents());
  }
}

void BufferOfCooccurrences::CalculateAndWritePpmi(const std::string mode, const long double n) {
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

double BufferOfCooccurrences::GetTokenFreq(const std::string& mode, const int token_id) const {
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
