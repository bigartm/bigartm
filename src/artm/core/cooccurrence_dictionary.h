// Copyright 2018, Additive Regularization of Topic Models.

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
#define BATCH 0
#define OUTPUT_FILE 1
#define TOKEN_NOT_FOUND -1

namespace artm {
namespace core {

struct CoocInfo {
  int second_token_id;
  unsigned long long cooc_tf;
  unsigned cooc_df;
};

// Data in Cooccurrence batches are stored in cells
// Every cell refers to its first token id and holds info about tokens that co-occur with it
// You need firstly to read cell header then records
struct Cell {
  Cell(int first_token_id = -1, unsigned num_of_records = 0) : 
       first_token_id(first_token_id), num_of_records(num_of_records) { }
  unsigned long long GetCoocFromCell(const std::string& mode, const unsigned record_pos) const {
    if (mode == "tf") {
      return records[record_pos].cooc_tf;
    } else {
      return records[record_pos].cooc_df;
    }
  }
  int first_token_id;
  unsigned num_of_records;  // when cell is read, it's necessary to know how many triples to read
  std::vector<CoocInfo> records;
};

struct TokenInfo {
  TokenInfo() : num_of_documents_token_occured_in(0), num_of_pairs_token_occured_in(0) { }
  unsigned num_of_documents_token_occured_in;
  unsigned long long num_of_pairs_token_occured_in;
};

class CooccurrenceDictionary;
class Vocab;
class CooccurrenceStatisticsHolder;
class CooccurrenceBatch;
class ResultingBufferOfCooccurrences;

class Vocab {
 friend class CooccurrenceDictionary;
 friend class ResultingBufferOfCooccurrences;
 public:
  struct TokenModality {
    TokenModality() { }
    TokenModality(const std::string token_str, const std::string modality) : token_str(token_str), modality(modality) { }
    std::string token_str;
    std::string modality;
  };
 private:
  Vocab(const std::string& path_to_vocab);
  std::string MakeKey(const std::string& token_str, const std::string& modality) const;
  int FindTokenId(const std::string& token_str, const std::string& modality) const;
  TokenModality FindTokenStr(const int token_id) const;

  std::unordered_map<std::string, int> token_map_;  // token|modality -> token_id
  std::unordered_map<int, TokenModality> inverse_token_map_;  // token_id -> (token, modality)
};

class CooccurrenceDictionary {
 public:
  CooccurrenceDictionary(const unsigned window_width, const unsigned cooc_min_tf,
      const unsigned cooc_min_df, const std::string& path_to_vocab,
      const std::string& path_to_vw, const std::string& cooc_tf_file_path,
      const std::string& cooc_df_file_path, const std::string& ppmi_tf_file_path,
      const std::string& ppmi_df_file_path, const int num_of_cpu, const unsigned doc_per_cooc_batch);
  unsigned VocabSize() const;
  void ReadVowpalWabbit();
  std::vector<std::string> ReadPortionOfDocuments(std::mutex& read_lock, std::ifstream& vowpal_wabbit_doc);
  unsigned CooccurrenceBatchesQuantity() const;
  ResultingBufferOfCooccurrences ReadAndMergeCooccurrenceBatches();
  ~CooccurrenceDictionary();
 private:
  std::string CreateFileInBatchDir() const;
  void UploadOnDisk(CooccurrenceStatisticsHolder& cooc_stat_holder);
  void FirstStageOfMerging();
  ResultingBufferOfCooccurrences SecondStageOfMerging(std::vector<std::unique_ptr<CooccurrenceBatch>>& intermediate_batches);
  void KWayMerge(ResultingBufferOfCooccurrences& res, const int mode,
                 std::vector<std::unique_ptr<CooccurrenceBatch>>& vector_of_batches,
                 CooccurrenceBatch& out_batch, std::mutex& open_file_lock);
  CooccurrenceBatch* CreateNewCooccurrenceBatch() const;
  void OpenBatchInputFile(CooccurrenceBatch& batch);
  void OpenBatchOutputFile(CooccurrenceBatch& batch);
  bool IsOpenBatchInputFile(const CooccurrenceBatch& batch) const;
  void CloseBatchInputFile(CooccurrenceBatch& batch);
  void CloseBatchOutputFile(CooccurrenceBatch& batch);

  const unsigned window_width_;
  const unsigned cooc_min_tf_;
  const unsigned cooc_min_df_;
  const std::string path_to_vw_;
  mutable std::string cooc_tf_file_path_;
  mutable std::string cooc_df_file_path_;
  const std::string ppmi_tf_file_path_;
  const std::string ppmi_df_file_path_;
  bool calculate_cooc_tf_;
  bool calculate_cooc_df_;
  bool calculate_ppmi_tf_;
  bool calculate_ppmi_df_;
  bool calc_symetric_cooc_;
  Vocab vocab_; // Holds mapping tokens to their indices
  std::vector<TokenInfo> token_statistics_; // index here is token_id which can be hound in vocab_dictionary
  std::string path_to_batches_;
  std::vector<std::unique_ptr<CooccurrenceBatch>> vector_of_batches_;
  unsigned open_files_counter_;
  unsigned max_num_of_open_files_;
  unsigned long long total_num_of_pairs_;
  unsigned total_num_of_documents_;
  unsigned doc_per_cooc_batch_;
  unsigned num_of_cpu_;
};

class CooccurrenceStatisticsHolder {
 friend class CooccurrenceDictionary;
 public:
  struct FirstToken;
  struct SecondTokenAndCooccurrence;
  void SavePairOfTokens(const int first_token_id, const int second_token_id, const unsigned doc_id, const double weight = 1);
 private:
  // Here's two-level structure storage_
  // Vector was chosen because it has a very low memory overhead
  std::map<int, FirstToken> storage_;
};

struct CooccurrenceStatisticsHolder::FirstToken {
  std::map<int, SecondTokenAndCooccurrence> second_token_reference;
};

struct CooccurrenceStatisticsHolder::SecondTokenAndCooccurrence {
  SecondTokenAndCooccurrence(const unsigned doc_id, const unsigned long long cooc_tf = 1) :
                             last_doc_id(doc_id), cooc_tf(cooc_tf), cooc_df(1) { }
  // When a new pair comes, this field is checked and if current doc_id isn't
  // equal to previous cooc_df should be incremented
  unsigned last_doc_id; // id of the last document where the pair occurred
  unsigned long long cooc_tf;
  unsigned cooc_df;
};

// Cooccurrence Batch is an intermidiate buffer between other data in RAM and
// a spesific file stored on disc. This buffer holds only one cell at a time.
// Also it's a wrapper around a ifstream and ofstream of an external file
class CooccurrenceBatch: private boost::noncopyable {
 friend class CooccurrenceDictionary;
 friend class ResultingBufferOfCooccurrences;
 public:
  struct CoocBatchComparator;
  void FormNewCell(const std::map<int, CooccurrenceStatisticsHolder::FirstToken>::iterator& cooc_stat_node);
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

struct CooccurrenceBatch::CoocBatchComparator {
  bool operator()(const std::unique_ptr<CooccurrenceBatch>& left,
                  const std::unique_ptr<CooccurrenceBatch>& right) const {
    return left->cell_.first_token_id > right->cell_.first_token_id;
  }
};

class ResultingBufferOfCooccurrences {  // ToDo (MichaelSolotky): make inharitance from CooccurrenceDictionary
 friend class CooccurrenceDictionary;
 public:
  void CalculatePpmi();
 private:
  ResultingBufferOfCooccurrences(
      std::vector<TokenInfo>& token_statistics_,
      Vocab& vocab, 
      const unsigned cooc_min_tf = 0,
      const unsigned cooc_min_df = 0,
      const unsigned num_of_cpu = 1,
      const unsigned long long total_num_of_pairs = 0,
      const unsigned total_num_of_documents = 0,
      const bool calculate_cooc_tf = false,
      const bool calculate_cooc_df = false,
      const bool calculate_ppmi_tf = false,
      const bool calculate_ppmi_df = false,
      const bool calc_symetric_cooc = false,
      const std::string& cooc_tf_file_path = "",
      const std::string& cooc_df_file_path = "",
      const std::string& ppmi_tf_file_path = "",
      const std::string& ppmi_df_file_path = "");
  void OpenAndCheckInputFile(std::ifstream& ifile, const std::string& path);
  void OpenAndCheckOutputFile(std::ofstream& ofile, const std::string& path);
  void MergeWithExistingCell(const CooccurrenceBatch& batch);
  void CalculateTFStatistics();
  void WriteCoocFromCell(const std::string mode, const unsigned cooc_min);  // Output file formats are defined here
  unsigned long long GetCoocFromCell(const std::string& mode, const unsigned record_pos) const;
  void CalculateAndWritePpmi(const std::string mode, const long double n, 
                      std::ifstream& cooc_dict_in, std::ofstream& ppmi_dict_out);
  double GetTokenFreq(const std::string& mode, const int token_id) const;

  std::vector<TokenInfo>& token_statistics_;
  Vocab& vocab_;  // Holds mapping tokens to their indices
  const unsigned cooc_min_tf_;
  const unsigned cooc_min_df_;
  const unsigned num_of_cpu_;
  const unsigned long long total_num_of_pairs_;
  const unsigned total_num_of_documents_;
  unsigned open_files_in_buf_;
  const bool calculate_cooc_tf_;
  const bool calculate_cooc_df_;
  const bool calculate_ppmi_tf_;
  const bool calculate_ppmi_df_;
  const bool calc_symetric_cooc_;
  std::ifstream cooc_tf_dict_in_;
  std::ofstream cooc_tf_dict_out_;
  std::ifstream cooc_df_dict_in_;
  std::ofstream cooc_df_dict_out_;
  std::ofstream ppmi_tf_dict_;
  std::ofstream ppmi_df_dict_;
  Cell cell_;
};

}  // namespace core
}  // namespace artm
// vim: set ts=2 sw=2:
