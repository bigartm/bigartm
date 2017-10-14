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
        total_num_of_pairs_(0), total_num_of_documents_(0),
        num_of_threads_(num_of_threads) {
  // This function works as follows:
  // 1. Get content from a vocab file and put it in dictionary
  // 2. Read Vowpal Wabbit file by portions, calculate co-occurrences for
  // every portion and save it (cooccurrence batch) on external storage
  // 3. Read from external storage all the cooccurrence batches piece by
  // piece and create resulting file with all co-occurrences

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
  open_files_counter_ = 0;
  max_num_of_open_files_ = 500;
  if (num_of_threads_ == -1) {
    num_of_threads_ = std::thread::hardware_concurrency();
    if (num_of_threads_ == 0) {
      num_of_threads_ = 1;
    }
  }
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
  if (!vocab.is_open()) {
    throw std::invalid_argument("Failed to open vocab");
  }
  int last_token_id = 1;
  std::string str;

  while (true) {
    getline(vocab, str);
    if (vocab.eof()) {
      break;
    }

    boost::algorithm::trim(str);
    std::vector<std::string> strs;
    boost::split(strs, str, boost::is_any_of(" "));
    if (!strs[0].empty()) {
      if (strs.size() == 1 || strcmp(strs[1].c_str(), "@default_class") == 0) {
        vocab_dictionary_.insert(std::make_pair(strs[0], last_token_id++));
      }
    }
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

  std::cout << "Step 1: creation of cooccurrence batches\n";
  std::ifstream vowpal_wabbit_doc(path_to_vw_, std::ios::in);
  if (!vowpal_wabbit_doc.is_open()) {
    throw std::invalid_argument("Failed to open vocab");
  }
  std::mutex read_lock;

  //unsigned critical_num_of_documents = 5000;
  //unsigned documents_processed = 0;
  auto func = [&]() {
    while (true) {
      std::vector<std::string> portion;

      {
        std::lock_guard<std::mutex> guard(read_lock);
        if (vowpal_wabbit_doc.eof()) {
          return;
	}

        std::string str;
        while (portion.size() < items_per_batch_) {
          getline(vowpal_wabbit_doc, str);
          if (vowpal_wabbit_doc.eof()) {
            break;
	  }
          portion.push_back(str);
        }
      }

      if (portion.size() == 0) {
        continue;
      }
      //documents_processed += portion.size();
      /*if (documents_processed >= critical_num_of_documents)
        break;*/

      // First elem in external map is first_token_id, in internal it's
      // second_token_id
      CoocMap cooc_map;

      total_num_of_documents_ += portion.size();
      // When the document is processed (element of vector portion),
      // memory for it can be freed by calling pop_back() from vector
      // (string will be popped)
      for (; portion.size() != 0; portion.pop_back()) {
        std::vector<std::string> doc;
        boost::split(doc, portion.back(), boost::is_any_of(" \t\r"));
        if (doc.size() <= 1) {
          continue;
	}
        const int default_class = 0;
        const int unusual_class = 1;
        int current_class = default_class;
        for (unsigned j = 1; j < doc.size() - 1; ++j) {
          if (doc[j][0] == '|') {
            if (strcmp(doc[j].c_str(), "|@default_class") == 0) {
              current_class = default_class;
            } else {
              current_class = unusual_class;
            }
            continue;
          }
          if (current_class != default_class) {
            continue;
          }

          auto first_token = vocab_dictionary_.find(doc[j]);
          if (first_token == vocab_dictionary_.end()) {
            continue;
          }

          {
            int current_class = default_class;
            int first_token_id = first_token->second;
            unsigned not_a_word_counter = 0;
            // if there are some words beginnig on '|' in a text the window
            // should be extended
            for (unsigned k = 1; k <= window_width_ + not_a_word_counter && j + k < doc.size(); ++k) {
              // ToDo: write macro here
              if (doc[j + k][0] == '|') {
                if (strcmp(doc[j + k].c_str(), "|@default_class") == 0) {
                  current_class = default_class;
                } else {
                  current_class = unusual_class;
                }
                ++not_a_word_counter;
                continue;
              }
              if (current_class != default_class) {
                continue;
              }

              auto second_token = vocab_dictionary_.find(doc[j + k]);
              if (second_token == vocab_dictionary_.end()) {
                continue;
              }
              int second_token_id = second_token->second;

              SavePairOfTokens(first_token_id, second_token_id, portion.size(), cooc_map);
              SavePairOfTokens(second_token_id, first_token_id, portion.size(), cooc_map);
              total_num_of_pairs_ += 2;
            }
          }
        }
      }

      if (!cooc_map.empty()) {
        UploadCooccurrenceBatchOnDisk(cooc_map);
      }
      //std::cout << documents_processed << " documents proccessed\n";
    }
  };

  std::vector<std::shared_future<void>> tasks;
  for (int i = 0; i < num_of_threads_; ++i) {
    tasks.push_back(std::move(std::async(std::launch::async, func)));
  }
  for (int i = 0; i < num_of_threads_; ++i) {
    tasks[i].get();
  }
  std::cout << "Cooccurrence batches have been created\n";
}

int CooccurrenceDictionary::CooccurrenceBatchQuantity() {
  return vector_of_batches_.size();
}

void CooccurrenceDictionary::ReadAndMergeCooccurrenceBatches() {
  std::cout << "Step 2: merging Batches\n";

  // This buffer won't hold more than 1 element, because another element
  // means another first_token_id => all the data linked with current
  // first_token_id can be filtered and be ready to be written in
  // resulting file.
  ResultingBuffer res(cooc_min_tf_, cooc_min_df_, calculate_tf_cooc_,
          calculate_df_cooc_, calculate_tf_ppmi_, calculate_df_ppmi_,
          calculate_ppmi_, total_num_of_pairs_, total_num_of_documents_,
          cooc_tf_file_path_, cooc_df_file_path_, ppmi_tf_file_path_, ppmi_df_file_path_);
  // ToDo: invent another mothod to add and subtract this number
  open_files_counter_ += res.open_files_in_buf_;

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

  // Standard k-way merge as external sort
  while (!vector_of_batches_.empty()) {
    // It's guaranteed that batches aren't empty (look ParseVowpalWabbit func)
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
  if (res.cell_.records.size() != 0) {
    res.PopPreviousContent();
  }
  // Files are close in order to really push data in files
  if (calculate_tf_cooc_) {
    res.cooc_tf_dict_out_.close();
  }

  if (calculate_df_cooc_) {
    res.cooc_df_dict_out_.close();
  }

  std::cout << "Batches have been merged\n";
  if (calculate_ppmi_) {
    std::cout << "Step 3: start calculation ppmi\n";
    res.CalculateAndWritePpmi();
  }
  std::cout << "Ppmi's have been calculated\n";
}

// ToDo: finish for mac, win, unices not caught below
// Note: user has to have some special headers on system (look in includes of)
// cooccurrence_dictionary.h
int CooccurrenceDictionary::SetItemsPerBatch() {
  // Here is a tool that allows to define an otimal value of documents that
  // should be load in ram. It depends on size of ram, window width and
  // num of threads (because every thread holds its batch of documents)
  const int custom_value = 6250;
  const int default_value = 10000;
  const double percent_of_ram = 0.4;
  const long long std_ram_size = 4025409536; // 4 Gb
  const int std_window_width = 10;
  const int std_num_of_threads = 2;
  long long totalram;
#if defined(_WIN32)
  return default_value;
#elif defined(__APPLE__)
  #if defined(TARGET_OS_MAC)
    return default_value;
  #endif
#elif defined(__linux__) || defined(__linux) || defined(linux)
  struct sysinfo ex;
  sysinfo(&ex);
  totalram = ex.totalram;
#elif defined(__unix__) // all unices not caught above
  return default_value;
#else // other platforms
  return default_value;
#endif
  return static_cast<double>(std_window_width) / window_width_ *
      totalram / std_ram_size * std_num_of_threads / num_of_threads_ *
      custom_value * percent_of_ram;
}

void CooccurrenceDictionary::SavePairOfTokens(const int first_token_id,
        const int second_token_id, const int doc_id, CoocMap& cooc_map) {
  auto map_record = cooc_map.find(first_token_id);
  if (map_record == cooc_map.end()) {
    AddInCoocMap(first_token_id, second_token_id, doc_id, cooc_map);
  } else {
    ModifyCoocMapNode(second_token_id, doc_id, map_record->second);
  }
}

void CooccurrenceDictionary::AddInCoocMap(const int first_token_id,
        const int second_token_id, const int doc_id, CoocMap& cooc_map) {
  FirstTokenInfo new_first_token(doc_id);
  CooccurrenceInfo new_cooc_info(doc_id);
  SecondTokenInfo new_second_token;
  new_second_token.insert(std::pair<int, CooccurrenceInfo>(second_token_id, new_cooc_info));
  cooc_map.insert(std::make_pair(first_token_id, std::make_pair(new_first_token, new_second_token)));
}

void CooccurrenceDictionary::ModifyCoocMapNode(const int second_token_id,
        const int doc_id, std::pair<FirstTokenInfo, SecondTokenInfo>& map_info) {
  if (std::get<FIRST_TOKEN_INFO>(map_info).prev_doc_id != doc_id) {
    std::get<FIRST_TOKEN_INFO>(map_info).prev_doc_id = doc_id;
    ++(std::get<FIRST_TOKEN_INFO>(map_info).num_of_documents);
  }
  SecondTokenInfo& map_node = std::get<SECOND_TOKEN_INFO>(map_info);
  auto iter = map_node.find(second_token_id);
  if (iter == map_node.end()) {
    CooccurrenceInfo new_cooc_info(doc_id);
    map_node.insert(std::pair<int, CooccurrenceInfo>(second_token_id, new_cooc_info));
  } else {
    ++(std::get<COOCCURRENCE_INFO>(*iter).cooc_tf);
    if (std::get<COOCCURRENCE_INFO>(*iter).prev_doc_id != doc_id) {
      std::get<COOCCURRENCE_INFO>(*iter).prev_doc_id = doc_id;
      ++(std::get<COOCCURRENCE_INFO>(*iter).cooc_df);
    }
  }
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

CooccurrenceBatch* CooccurrenceDictionary::CreateNewCooccurrenceBatch() {
  return new CooccurrenceBatch(path_to_batches_);
}

void CooccurrenceDictionary::OpenBatchInputFile(CooccurrenceBatch& batch) {
  assert(open_files_counter_ < max_num_of_open_files_);
  ++open_files_counter_;
  batch.in_batch_.open(batch.filename_, std::ios::in);
  batch.in_batch_.seekg(batch.in_batch_offset_);
}

void CooccurrenceDictionary::OpenBatchOutputFile(CooccurrenceBatch& batch) {
  assert(open_files_counter_ < max_num_of_open_files_);
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

// ********************Methods of class CoccurrenceBatch**************

void CooccurrenceBatch::FormNewCell(const CoocMap::iterator& map_node) {
  // Every cooccurrence batch is divided into cells as folowing:
  // Different cells have different first token id values.
  // One cell contain records with euqal first token id
  // One cooccurrence batch can't hold 2 or more cells in ram simultaneously
  // Other cells are stored in output file
  cell_.first_token_id = std::get<FIRST_TOKEN_ID>(*map_node);
  std::pair<FirstTokenInfo, SecondTokenInfo>& map_info = std::get<MAP_INFO>(*map_node);
  cell_.num_of_documents = std::get<FIRST_TOKEN_INFO>(map_info).num_of_documents;
  SecondTokenInfo& second_token_info = std::get<SECOND_TOKEN_INFO>(map_info);
  cell_.num_of_records = second_token_info.size();
  cell_.records.resize(cell_.num_of_records);
  int i = 0;
  for (auto iter = second_token_info.begin(); iter != second_token_info.end(); ++iter, ++i) {
    cell_.records[i].second_token_id = std::get<SECOND_TOKEN_ID>(*iter);
    CooccurrenceInfo& cooc_info = std::get<COOCCURRENCE_INFO>(*iter);
    cell_.records[i].cooc_tf = cooc_info.cooc_tf;
    cell_.records[i].cooc_df = cooc_info.cooc_df;
  }
}

// Cells are written in following form: first line consists of first token id
// and num of triples
// the second line consists of numbers triples, which are separeted with a
// space and numbers in these triples are separeted the same
void CooccurrenceBatch::WriteCell() {
  // stringstream is used for fast bufferized i/o operations
  std::stringstream ss;
  ss << cell_.first_token_id << ' ';
  ss << cell_.num_of_documents << ' ';
  ss << cell_.num_of_records << std::endl;
  for (unsigned i = 0; i < cell_.records.size(); ++i) {
    ss << cell_.records[i].second_token_id << ' ';
    ss << cell_.records[i].cooc_tf << ' ';
    ss << cell_.records[i].cooc_df << ' ';
  }
  ss << std::endl;
  out_batch_ << ss.str();
}

bool CooccurrenceBatch::ReadCellHeader() {
  // stringstream is used for fast bufferized i/o operations
  std::string str;
  getline(in_batch_, str);
  std::stringstream ss(str);
  ss >> cell_.first_token_id;
  ss >> cell_.num_of_documents;
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

bool CooccurrenceBatch::ReadCell() {
  if (ReadCellHeader()) {
    ReadRecords();
    return true;
  }
  return false;
}

CooccurrenceBatch::CooccurrenceBatch(const std::string& path_to_batches) {
  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  fs::path batch(boost::lexical_cast<std::string>(uuid));
  fs::path full_filename = fs::path(path_to_batches) / batch;
  filename_ = full_filename.string();
  in_batch_offset_ = 0;
}

// ********************Methods of class ResultingBuffer**************

ResultingBuffer::ResultingBuffer(const int cooc_min_tf, const int cooc_min_df,
    const bool calculate_cooc_tf, const bool calculate_cooc_df,
    const bool calculate_ppmi_tf, const bool calculate_ppmi_df,
    const bool calculate_ppmi, const long long total_num_of_pairs,
    const int total_num_of_documents,
    const std::string& cooc_tf_file_path,
    const std::string& cooc_df_file_path,
    const std::string& ppmi_tf_file_path,
    const std::string& ppmi_df_file_path) : cooc_min_tf_(cooc_min_tf),
        cooc_min_df_(cooc_min_df), calculate_cooc_tf_(calculate_cooc_tf),
        calculate_cooc_df_(calculate_cooc_df), calculate_ppmi_tf_(calculate_ppmi_tf),
        calculate_ppmi_df_(calculate_ppmi_df), calculate_ppmi_(calculate_ppmi),
        total_num_of_pairs_(total_num_of_pairs),
        total_num_of_documents_(total_num_of_documents),
        output_buf_size_(8500), open_files_in_buf_(0) {

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

void ResultingBuffer::OpenAndCheckInputFile(std::ifstream& ifile, const std::string& path) {
  ifile.open(path, std::ios::in);
  if (!ifile.good()) {
    throw std::invalid_argument("Failed to create a file in the working directory");
  }
  ++open_files_in_buf_;
}

void ResultingBuffer::OpenAndCheckOutputFile(std::ofstream& ofile, const std::string& path) {
  ofile.open(path, std::ios::out);
  if (!ofile.good()) {
    throw std::invalid_argument("Failed to create a file in the working directory");
  }
  ++open_files_in_buf_;
}

void ResultingBuffer::AddInBuffer(const CooccurrenceBatch& batch) {
  if (cell_.first_token_id == batch.cell_.first_token_id) {
    MergeWithExistingCell(batch);
  } else {
    PopPreviousContent();
    cell_ = batch.cell_;
  }
}

void ResultingBuffer::MergeWithExistingCell(const CooccurrenceBatch& batch) {
  std::vector<CoocTriple> old_vector = cell_.records;
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

void ResultingBuffer::PopPreviousContent() {
  // stringstream is used for fast bufferized i/o operations
  std::stringstream output_buf_tf;
  std::stringstream output_buf_df;
  PpmiCountersValues n_u;
  for (unsigned i = 0; i < cell_.records.size(); ++i) {
    if (calculate_cooc_tf_ && cell_.records[i].cooc_tf >= cooc_min_tf_) {
      if (cell_.first_token_id != cell_.records[i].second_token_id) {
        output_buf_tf << cell_.first_token_id << ' ' << cell_.records[i].second_token_id << ' ' << cell_.records[i].cooc_tf << std::endl;
      }
      n_u.n_u_tf += cell_.records[i].cooc_tf;
    }

    if (output_buf_tf.tellg() > output_buf_size_) {
      cooc_tf_dict_out_ << output_buf_tf.str();
    }

    if (calculate_cooc_df_ && cell_.records[i].cooc_df >= cooc_min_df_) {
      if (cell_.first_token_id != cell_.records[i].second_token_id) {
        output_buf_df << cell_.first_token_id << ' ' << cell_.records[i].second_token_id << ' ' << cell_.records[i].cooc_df << std::endl;
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

  if (n_u.n_u_tf != 0) {
    n_u.n_u_df = cell_.num_of_documents;
    ppmi_counters_.insert(std::make_pair(cell_.first_token_id, n_u));
  }
  // It's importants after pop to set size = 0, because this value will be checked later
  cell_.records.resize(0);
  //std::cout << "Token " << cell_.first_token_id << " has been proccessed\n";
}

// ToDo: erase duplications of code
void ResultingBuffer::CalculateAndWritePpmi() {
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
          ppmi_counters_[first_token_id].n_u_df) /
          (ppmi_counters_[second_token_id].n_u_df / static_cast<double>(cooc_df));
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
