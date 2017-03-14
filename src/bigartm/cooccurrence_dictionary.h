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

struct FirstTokenInfo {
  int doc_num;
  int prev_doc_id;
};

struct CooccurrenceInfo {
  int cooc_tf;
  int cooc_df;
  int prev_doc_id;
};

struct InputInfo {
  int cooc_tf;
  int cooc_df;
};

struct OutputInfo {
  long long cooc_value;
  double ppmi;
};

typedef std::map<int, CooccurrenceInfo> SecondTokenInfo;
typedef std::map<int, std::pair<FirstTokenInfo, SecondTokenInfo>> CoocMap;
typedef std::list<std::pair<int, OutputInfo>> OutputRecords;
// Here in pair <int, OutputRecords> int is number of documents where the
// folowing token occurred
// list<pair<first_token_id, pair<num_of_documents, OutputRecords>>>
typedef std::list<std::pair<int, std::pair<int, OutputRecords>>> OutputList;

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
  void UploadCooccurrenceBatchOnDisk(CoocMap& cooc_map);
  CooccurrenceInfo FormInitialCoocInfo(int doc_id);
  FirstTokenInfo FormInitialFirstTokenInfo(int doc_id);
  void SavePairOfTokens(int first_token_id, int second_token_id, int doc_id,
          CoocMap& cooc_map);
  void AddInCoocMap(int first_token_id, int second_token_id, int doc_id,
          CoocMap& cooc_map);
  void ModifyCoocMapNode(int second_token_id, int doc_id,
          std::pair<FirstTokenInfo, SecondTokenInfo>& map_node);
  void SaveNumberOfDocuments(int first_token_id);
  CooccurrenceBatch* CreateNewCooccurrenceBatch();
  void OpenBatchInputFile(CooccurrenceBatch& batch);
  void OpenBatchOutputFile(CooccurrenceBatch& batch);
  bool IsOpenBatchInputFile(CooccurrenceBatch& batch);
  void CloseBatchInputFile(CooccurrenceBatch& batch);
  void CloseBatchOutputFile(CooccurrenceBatch& batch);

  const int window_width_;
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
  int items_per_batch_;
  long long total_num_of_pairs_;
  int total_num_of_documents_;
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

  struct Cell {
    int first_token_id;
    int num_of_documents; // number of documents where first token occurred
    int num_of_records;
    std::vector<std::pair<int, InputInfo>> batch_records;
    OutputRecords tf_records;
    OutputRecords df_records;
  };
  Cell current_cell_;
  std::ifstream in_batch_;
  std::ofstream out_batch_;
  std::string filename_;
  long in_batch_offset_;
};

class ResultingBuffer {
 friend class CooccurrenceDictionary;
 private:
  ResultingBuffer(const int cooc_min_tf, const int cooc_min_df,
      const bool calculate_tf_cooc, const bool calculate_df_cooc,
      const bool calculate_tf_ppmi, const bool calculate_df_ppmi,
      const bool calculate_ppmi_, const long long total_num_of_pairs,
      const int total_num_of_documents,
      const std::string& cooc_tf_file_path,
      const std::string& cooc_df_file_path,
      const std::string& ppmi_tf_file_path,
      const std::string& ppmi_df_file_path);
  void AddInBuffer(const CooccurrenceBatch& batch);
  void MergeWithExistingCell(OutputList& vector_of_records,
          const OutputRecords& batch_records,
          const int num_of_documents_from_batch_cell);
  void CheckPreviousCell(OutputList& list_of_records, int cooc_min_value);
  void AddNewCellInBuffer(OutputList& list_of_records,
          const OutputRecords& batch_records, const int first_token_id,
          const int num_of_documents);
  void BuildFreqDictionary();
  void CalculateTfPpmi();
  void CalculateDfPpmi();
  void WriteCoocInResultingFile(OutputList& list_of_records, std::ofstream& cooc_dict);
  void WritePpmiInResultingFile(OutputList& list_of_records, std::ofstream& ppmi_dict);

  const int cooc_min_tf_;
  const int cooc_min_df_;
  const bool calculate_tf_cooc_;
  const bool calculate_df_cooc_;
  const bool calculate_tf_ppmi_;
  const bool calculate_df_ppmi_;
  const bool calculate_ppmi_;
  const long long total_num_of_pairs_;
  const int total_num_of_documents_;
  std::ofstream cooc_tf_dict_;
  std::ofstream cooc_df_dict_;
  std::ofstream ppmi_tf_dict_;
  std::ofstream ppmi_df_dict_;
  OutputList list_of_tf_records_;
  OutputList list_of_df_records_;
  std::unordered_map<int, std::pair<long long, int>> freq_dictionary_;
  const unsigned filebuf_size_;
};
