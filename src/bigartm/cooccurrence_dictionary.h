// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <sstream>
#include <iomanip>
#include <memory>
#include <mutex>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/utility.hpp"

#define FIRST_TOKEN_ID 0
#define FIRST_TOKEN_INFO 0
#define SECOND_TOKEN_ID 0
#define SECOND_TOKEN_INFO 1
#define COOCCURRENCE_INFO 1
#define MAP_INFO 1
#define NOT_FOUND -1

struct CoocInfo {
  int second_token_id;
  long long cooc_tf;
  int cooc_df;
};

struct PpmiCountersValues {
  PpmiCountersValues() : n_u_tf(0), n_u_df(0) { }
  long long n_u_tf;
  int n_u_df;
};

// Data in Coccurrence batches are stored in cells
// Every cell refers to its first token id and holds info about tokens that co-occur with it
// You need firstly to read cell header then records
struct Cell {
  Cell() : first_token_id(-1), num_of_records(0) { }
  int first_token_id;
  unsigned num_of_records; // when cell is read, it's necessary to know how many triples to read
  std::vector<CoocInfo> records;
};

struct TokenInfo {
  TokenInfo() : num_of_pairs_token_occured_in(0), num_of_documents_token_occured_in(0) { }
  unsigned num_of_pairs_token_occured_in;
  unsigned num_of_documents_token_occured_in;
};

enum modality_label {
  DEFAULT_CLASS,
  UNUSUAL_CLASS
};

class CooccurrenceDictionary;
class CooccurrenceStatisticsHolder;
class CooccurrenceBatch;
class ResultingBufferOfCoccurrences;

class CooccurrenceDictionary {
 public:
  CooccurrenceDictionary(const int window_width, const int cooc_min_tf,
      const int cooc_min_df, const std::string& path_to_vocab,
      const std::string& path_to_vw, const std::string& cooc_tf_file_path,
      const std::string& cooc_df_file_path,
      const std::string& ppmi_tf_file_path,
      const std::string& ppmi_df_file_path, const int num_of_threads);
  void FetchVocab();
  int  VocabDictionarySize();
  void ReadVowpalWabbit();
  std::vector<std::string> ReadPortionOfDocuments(std::mutex& read_lock, std::ifstream& vowpal_wabbit_doc);
  int  SetModalityLabel(std::string& modality_label);
  void UploadOnDisk(CooccurrenceStatisticsHolder& cooc_stat_holder);
  int  CooccurrenceBatchesQuantity();
  void ReadAndMergeCooccurrenceBatches();
  ~CooccurrenceDictionary();
 private:
  CooccurrenceBatch* CreateNewCooccurrenceBatch();
  void OpenBatchInputFile(CooccurrenceBatch& batch);
  void OpenBatchOutputFile(CooccurrenceBatch& batch);
  bool IsOpenBatchInputFile(CooccurrenceBatch& batch);
  void CloseBatchInputFile(CooccurrenceBatch& batch);
  void CloseBatchOutputFile(CooccurrenceBatch& batch);

  const unsigned window_width_;
  const int cooc_min_tf_;
  const int cooc_min_df_;
  const std::string path_to_vocab_;
  const std::string path_to_vw_;
  const std::string cooc_tf_file_path_;
  const std::string cooc_df_file_path_;
  const std::string ppmi_tf_file_path_;
  const std::string ppmi_df_file_path_;
  bool calculate_tf_ppmi_;
  bool calculate_df_ppmi_;
  bool calculate_ppmi_;
  bool calculate_tf_cooc_;
  bool calculate_df_cooc_;
  std::unordered_map<std::string, int> vocab_dictionary_; // mapping tokens to their indices
  std::vector<TokenInfo> token_statistics_; // index here is token_id which can be hound in vocab_dictionary
  std::string path_to_batches_;
  std::vector<std::unique_ptr<CooccurrenceBatch>> vector_of_batches_;
  int open_files_counter_;
  int max_num_of_open_files_;
  unsigned total_num_of_pairs_;
  unsigned total_num_of_documents_;
  unsigned documents_per_batch_;
  int num_of_threads_;
};

class CooccurrenceStatisticsHolder {
 friend class CooccurrenceDictionary;
 public:
  struct FirstToken;
  struct SecondTokenAndCooccurrence;

  void SavePairOfTokens(int first_token_id, int second_token_id, unsigned doc_id);
  int  FindFirstToken(int first_token_id);
  int  FindSecondToken(std::vector<SecondTokenAndCooccurrence>& second_tokens, int second_token_id);
  bool IsEmpty();
  void SortFirstTokens();
  void SortSecondTokens();
 private:

  // Here's two-level structure storage_
  // Vector was chosen because it has a very low memory overhead
  std::vector<FirstToken> storage_;
};

struct CooccurrenceStatisticsHolder::FirstToken {
  FirstToken(int first_token_id) : first_token_id(first_token_id) { }
  int first_token_id;
  std::vector<SecondTokenAndCooccurrence> second_token_reference;
};

struct CooccurrenceStatisticsHolder::SecondTokenAndCooccurrence {
  SecondTokenAndCooccurrence(int second_token_id) : second_token_id(second_token_id), cooc_tf(1), cooc_df(1) { }
  int second_token_id;
  // When a new pair comes, this field is checked and if current doc_id isn't
  // equal to previous cooc_df should be incremented
  unsigned last_doc_id;
  unsigned cooc_tf;
  unsigned cooc_df;
};

// Cooccurrence Batch is an intermidiate buffer between other data in RAM and
// a spesific file stored on disc. This buffer holds only one cell at a time.
// Also it's a wrapper around a ifstream and ofstream of an external file
class CooccurrenceBatch: private boost::noncopyable {
 friend class CooccurrenceDictionary;
 friend class ResultingBufferOfCoccurrences;
 public:
  void FormNewCell(const std::vector<CooccurrenceStatisticsHolder::FirstToken>::iterator& cooc_stat_node);
  void WriteCell();
  bool ReadCellHeader();
  void ReadRecords();
  bool ReadCell(); // Initiates reading of a cell from a file (e.g. call of ReadCellHeader() and ReadRecords())
 private:
  CooccurrenceBatch(const std::string& path_to_batches);

  Cell cell_;
  std::ifstream in_batch_;
  std::ofstream out_batch_;
  std::string filename_;
  long in_batch_offset_;
};

class ResultingBufferOfCoccurrences {
 friend class CooccurrenceDictionary;
 private:
  ResultingBufferOfCoccurrences(const int cooc_min_tf, const int cooc_min_df,
      const bool calculate_cooc_tf, const bool calculate_cooc_df,
      const bool calculate_ppmi_tf, const bool calculate_ppmi_df,
      const bool calculate_ppmi, const long long total_num_of_pairs_,
      const int total_num_of_documents_,
      const std::string& cooc_tf_file_path,
      const std::string& cooc_df_file_path,
      const std::string& ppmi_tf_file_path,
      const std::string& ppmi_df_file_path,
      const std::vector<TokenInfo>& token_statistics_);
  void OpenAndCheckInputFile(std::ifstream& ifile, const std::string& path);
  void OpenAndCheckOutputFile(std::ofstream& ofile, const std::string& path);
  void AddInBuffer(const CooccurrenceBatch& batch);
  void MergeWithExistingCell(const CooccurrenceBatch& batch);
  void WriteCoocFromBufferInFile(); // output file format (of variety of formats) is difined here
  void CalculateAndWritePpmi();

  const int cooc_min_tf_;
  const int cooc_min_df_;
  const bool calculate_cooc_tf_;
  const bool calculate_cooc_df_;
  const bool calculate_ppmi_tf_;
  const bool calculate_ppmi_df_;
  const bool calculate_ppmi_;
  const long long total_num_of_pairs_;
  const int total_num_of_documents_;
  const int output_buf_size_;
  int open_files_in_buf_;
  std::ifstream cooc_tf_dict_in_;
  std::ofstream cooc_tf_dict_out_;
  std::ifstream cooc_df_dict_in_;
  std::ofstream cooc_df_dict_out_;
  std::ofstream ppmi_tf_dict_;
  std::ofstream ppmi_df_dict_;
  std::vector<TokenInfo> token_statistics_;

  Cell cell_;
  std::unordered_map<int, PpmiCountersValues> ppmi_counters_;
};
