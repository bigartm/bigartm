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

struct Triple {
  int cooc_value;
  int doc_quan;
  int second_token_id;
};

struct Pair {
  int cooc_value;
  int doc_quan;
};

struct CooccurrenceInfo {
  int cooc_value;
  int doc_quan;
  int prev_doc_id;
};

typedef std::map<int, CooccurrenceInfo> CoocMap;

class CooccurrenceBatch: private boost::noncopyable {
 friend class CooccurrenceDictionary;
 friend class ResultingBuffer;
 public:
  void FormNewCell(std::map<int, CoocMap>::iterator& map_node);
  void ReadRecords();
  void WriteCell();
  bool ReadCell();
  bool ReadCellHeader();
 private:
  CooccurrenceBatch(const std::string& path_to_batches);

  struct Cell {
    int first_token_id;
    int num_of_triples;
    std::vector<Triple> records;
  };
  Cell cell_;
  std::ifstream in_batch_;
  std::ofstream out_batch_;
  std::string filename_;
};

class ResultingBuffer {
 public:
  ResultingBuffer(const int min_tf, const int min_df,
          const std::string& cooc_tf_file_path,
          const std::string& cooc_df_file_path,
          const bool& cooc_tf_flag, const bool& cooc_df_flag);
  ~ResultingBuffer();
  void AddInBuffer(const CooccurrenceBatch& batch);
 private:
  void MergeWithExistingCell(const CooccurrenceBatch& batch);
  void PopPreviousContent();
  void AddNewCellInBuffer(const CooccurrenceBatch& batch);

  int cooc_min_tf_;
  int cooc_min_df_;
  int first_token_id_;
  std::vector<Triple> rec_;
  std::ofstream cooc_tf_dict_;
  std::ofstream cooc_df_dict_;
  bool calculate_cooc_tf_;
  bool calculate_cooc_df_;
};

class CooccurrenceDictionary {
 public:
  CooccurrenceDictionary(const std::string& vw, const std::string& vocab,
        const std::string& cooc_tf_file, const std::string& cooc_df_file,
        const int wind_width, const int min_tf, const int min_df,
        const int items_per_batch);
  ~CooccurrenceDictionary();
  void FetchVocab();
  int  VocabDictionarySize();
  void ReadVowpalWabbit();
  int  CooccurrenceBatchQuantity();
  void ReadAndMergeCooccurrenceBatches();
 private:
  void UploadCooccurrenceBatchOnDisk(std::map<int, CoocMap>& cooc);
  CooccurrenceInfo FormInitialCoocInfo(int doc_id);
  void AddInCoocMap(int first_token_id, int second_token_id, int doc_id,
          std::map<int, CoocMap>& cooc_map);
  void ModifyCoocMapNode(int second_token_id, int doc_id,
          CoocMap& map_node);
  void SavePairOfTokens(int first_token_id, int second_token_id, int doc_id,
          std::map<int, CoocMap>& cooc_map);
  CooccurrenceBatch* CreateNewCooccurrenceBatch();
  void OpenExistingBatchFile(CooccurrenceBatch& batch);
  void CloseExistingBatchFile(CooccurrenceBatch& batch);

  int window_width_;
  int cooc_min_df_;
  int cooc_min_tf_;
  std::string path_to_vocab_;
  std::string path_to_vw_;
  std::string cooc_tf_file_path_;
  std::string cooc_df_file_path_;
  bool calculate_tf_cooc_;
  bool calculate_df_cooc_;
  std::unordered_map<std::string, int> vocab_dictionary_;
  unsigned max_num_of_batches_;
  int items_per_batch_;
  std::string path_to_batches_;
  std::vector<std::unique_ptr<CooccurrenceBatch>> vector_of_batches_;
  int open_files_counter_;
};
