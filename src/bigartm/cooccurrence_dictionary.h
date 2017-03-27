// Copyright 2017, Additive Regularization of Topic Models.

//#include <cstdio>
#include <unordered_map>
#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <map>
#include <vector>
#include <sstream>
#include <iomanip>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/utility.hpp"

enum {
  MAX_NUM_OF_BATCHES = 1000,
  ITEMS_PER_BATCH = 10000,
};

struct Triple {
  double cooc_value;
  int doc_quan;
  int second_token_id;
};

struct Pair {
  double cooc_value;
  int doc_quan;
};

struct cooccurrence_info {
  double cooc_value;
  int doc_quan, prev_doc_id;
};

using namespace std;
using namespace boost;

class CooccurrenceBatch: private boost::noncopyable {
  struct Cell {
    int first_token_id;
    int num_of_triples;
    std::vector<Triple> records;
  };

  friend class BatchManager;
 private:
  std::ifstream in_batch;
  std::ofstream out_batch;

  CooccurrenceBatch(int batch_num, const char filemode, const string &disk_path);
 public:
  Cell cell;

  void FormNewCell(std::map<int, std::map<int, cooccurrence_info>>::iterator &map_node);

  int ReadCellHeader();

  void ReadRecords();

  int ReadCell();

  void WriteCell();
};

class BatchManager {
 private:
  int batch_quan;
  const int max_batch_quan = 1000;
  std::string path_to_batches;
 public:
  BatchManager();

  ~BatchManager();

  int GetBatchQuan();

  CooccurrenceBatch *CreateNewBatch();

  CooccurrenceBatch *OpenExistingBatch(int batch_num);
};

class ResultingBuffer {
 private:
  double cooc_min_tf;
  int cooc_min_df;
  int first_token_id;
  std::vector<Triple> rec;
  std::ofstream cooc_tf_dict;
  std::ofstream cooc_df_dict;
  bool calculate_cooc_tf;
  bool calculate_cooc_df;

  void MergeWithExistingCell(const CooccurrenceBatch *batch);

  // Note: here is cast to int and comparison of doubles
  void PopPreviousContent();

  void AddNewCellInBuffer(const CooccurrenceBatch *batch);
 public:
  ResultingBuffer(const double min_tf, const int min_df,
          const std::string &cooc_tf_file_path,
          const std::string &cooc_df_file_path, const bool &cooc_tf_flag,
          const bool &cooc_df_flag);

  ~ResultingBuffer();

  void AddInBuffer(const CooccurrenceBatch *batch);
};

class CooccurrenceDictionary {
 private:
  int window_width;
  int cooc_min_df;
  double cooc_min_tf;
  std::string path_to_vocab;
  std::string path_to_vw;
  std::string cooc_tf_file_path;
  std::string cooc_df_file_path;
  bool calculate_tf_cooc;
  bool calculate_df_cooc;
  BatchManager batch_manager;
  std::unordered_map<string, int> dictionary;

  void FetchVocab();

  void UploadBatchOnDisk(BatchManager &batch_manager,
        std::map<int, std::map<int, cooccurrence_info>> &cooc);

  cooccurrence_info FormInitialCoocInfo(int doc_id);

  void AddInCoocMap(int first_token_id, int second_token_id, int doc_id,
        std::map<int, std::map<int, cooccurrence_info>> &cooc_map);

  void ModifyCoocMapNode(int second_token_id, int doc_id,
          std::map<int, cooccurrence_info> &map_node);

  void SavePairOfTokens(int first_token_id, int second_token_id, int doc_id,
          std::map<int, std::map<int, cooccurrence_info>> &cooc_map);

  void ReadVowpalWabbit();

  void ReadAndMergeBatches();
 public:
  CooccurrenceDictionary(const std::string &vw, const std::string &vocab,
        const std::string &cooc_tf_file, const std::string &cooc_df_file,
        const int wind_width, const double min_tf, const int min_df);
};
