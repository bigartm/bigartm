// Copyright 2017, Additive Regularization of Topic Models.

#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <sstream>
#include <iomanip>
#include <memory>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/utility.hpp"

#define FIRST_TOKEN_ID 0
#define FIRST_TOKEN_INFO 0
#define SECOND_TOKEN_ID 0
#define SECOND_TOKEN_INFO 1
#define COOCCURRENCE_INFO 1
#define MAP_INFO 1
#define ABSOLUTE_VALUES 1
#define RESULTS 1

// Note: user has to have some of the headers below on his system
// ToDo: finish for mac, win, unices not caught below
#if defined(_WIN32)
#elif defined(__APPLE__)
  //#include "TargetConditionals.h"
#elif defined(__linux__) || defined(__linux) || defined(linux)
  #include <sys/sysinfo.h>
#elif defined(__unix__) // all unices not caught above
#endif
// https://stackoverflow.com/questions/5919996/how-to-detect-reliably-mac-os-x-ios-linux-windows-in-c-preprocessor

struct CoocTriple {
  int second_token_id;
  long long cooc_tf;
  int cooc_df;
};

struct CoocPair {
  long long cooc_tf;
  int cooc_df;
};

struct CooccurrenceInfo {
  CooccurrenceInfo(const int doc_id) : cooc_tf(1), cooc_df(1), prev_doc_id(doc_id) {}
  long long cooc_tf;
  int cooc_df;
  int prev_doc_id;
};

struct FirstTokenInfo {
  FirstTokenInfo(const int doc_id) : num_of_documents(1), prev_doc_id(doc_id) {}
  int num_of_documents;
  int prev_doc_id;
};

struct Results {
  Results() : cooc_tf(0), cooc_df(0), tf_ppmi(0.0), df_ppmi(0.0) {}
  long long cooc_tf;
  int cooc_df;
  double tf_ppmi;
  double df_ppmi;
};

struct AbsoluteValues {
  AbsoluteValues() : absolute_tf(0), absolute_df(0) {}
  long long absolute_tf;
  int absolute_df;
  std::unordered_map<int, Results> resulting_info;
};

struct Cell {
  Cell() : first_token_id(-1), num_of_documents(0), num_of_records(0) {}
  int first_token_id;
  int num_of_documents;
  unsigned num_of_records;
  std::vector<CoocTriple> records;
};

typedef std::map<int, CooccurrenceInfo> SecondTokenInfo;
typedef std::map<int, std::pair<FirstTokenInfo, SecondTokenInfo>> CoocMap;

class CooccurrenceDictionary;
class CooccurrenceBatch;
class ResultingBuffer;

class CooccurrenceDictionary {
 public:
  CooccurrenceDictionary(const int window_width, const int cooc_min_tf,
      const int cooc_min_df, const std::string& path_to_vocab,
      const std::string& path_to_vw, const std::string& cooc_tf_file_path,
      const std::string& cooc_df_file_path,
      const std::string& ppmi_tf_file_path,
      const std::string& ppmi_df_file_path);
  ~CooccurrenceDictionary();
  void FetchVocab();
  int  VocabDictionarySize();
  void ReadVowpalWabbit();
  int  CooccurrenceBatchQuantity();
  void ReadAndMergeCooccurrenceBatches();
 private:
  int SetItemsPerBatch();
  void SavePairOfTokens(const int first_token_id, const int second_token_id, const int doc_id,
          CoocMap& cooc_map);
  void AddInCoocMap(const int first_token_id, const int second_token_id, const int doc_id,
          CoocMap& cooc_map);
  void ModifyCoocMapNode(const int second_token_id, const int doc_id,
          std::pair<FirstTokenInfo, SecondTokenInfo>& map_info);
  void UploadCooccurrenceBatchOnDisk(CoocMap& cooc_map);
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
  bool write_tf_cooc_;
  bool write_df_cooc_;
  bool calculate_tf_ppmi_;
  bool calculate_df_ppmi_;
  bool calculate_ppmi_;
  bool calculate_tf_cooc_;
  bool calculate_df_cooc_;

  std::unordered_map<std::string, int> vocab_dictionary_;
  std::string path_to_batches_;
  std::vector<std::unique_ptr<CooccurrenceBatch>> vector_of_batches_;
  int open_files_counter_;
  int max_num_of_open_files_;
  int num_of_threads_;
  unsigned total_num_of_pairs_;
  unsigned total_num_of_documents_;
  unsigned items_per_batch_;
};

class CooccurrenceBatch: private boost::noncopyable {
 friend class CooccurrenceDictionary;
 friend class ResultingBuffer;
 public:
  void FormNewCell(const CoocMap::iterator& map_node);
  void WriteCell();
  bool ReadCellHeader();
  void ReadRecords();
  bool ReadCell();
 private:
  CooccurrenceBatch(const std::string& path_to_batches);

  Cell cell_;
  std::ifstream in_batch_;
  std::ofstream out_batch_;
  std::string filename_;
  long in_batch_offset_;
};

class ResultingBuffer {
 friend class CooccurrenceDictionary;
 private:
  ResultingBuffer(const int cooc_min_tf, const int cooc_min_df,
      const bool calculate_cooc_tf, const bool calculate_cooc_df,
      const bool calculate_ppmi_tf, const bool calculate_ppmi_df,
      const bool calculate_ppmi, const long long total_num_of_pairs_,
      const int total_num_of_documents_,
      const std::string& cooc_tf_file_path,
      const std::string& cooc_df_file_path,
      const std::string& ppmi_tf_file_path,
      const std::string& ppmi_df_file_path);
  void AddInBuffer(const CooccurrenceBatch& batch);
  void MergeWithExistingCell(const CooccurrenceBatch& batch);
  void PopPreviousContent();
  void CalculatePpmi();
  void WritePpmiInFile();

  const int cooc_min_tf_;
  const int cooc_min_df_;
  const bool calculate_cooc_tf_;
  const bool calculate_cooc_df_;
  const bool calculate_ppmi_tf_;
  const bool calculate_ppmi_df_;
  const bool calculate_ppmi_;
  const long long total_num_of_pairs_;
  const int total_num_of_documents_;
  std::ofstream cooc_tf_dict_;
  std::ofstream cooc_df_dict_;
  std::ofstream ppmi_tf_dict_;
  std::ofstream ppmi_df_dict_;

  Cell cell_;
  const int output_buf_size_;
  std::unordered_map<int, AbsoluteValues> resulting_hash_table_;
};