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
#include <mutex>  // NOLINT

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/utility.hpp"

#include "artm/core/collection_parser.h"

#include "artm/core/common.h"

namespace artm {
namespace core {

enum {
  TOKEN_NOT_FOUND = -1,
  BATCH = 0,
  OUTPUT_FILE = 1
};

struct CoocInfo {
  int second_token_id;
  int64_t cooc_tf;
  unsigned cooc_df;
};

// Data in Cooccurrence batches are stored in cells
// Every cell refers to its first token id and holds info about tokens that co-occur with it
// You need firstly to read cell header then records

struct Cell {
  explicit Cell(int first_token_id = -1, unsigned num_of_records = 0) :
       first_token_id(first_token_id), num_of_records(num_of_records) { }

  int64_t GetCoocFromCell(const std::string& mode, const unsigned record_pos) const {
    if (mode == TokenCoocFrequency) {
      return records[record_pos].cooc_tf;
    } else {
      return records[record_pos].cooc_df;
    }
  }
  int first_token_id;
  unsigned num_of_records;  // when cell is read, it's necessary to know how many triples to read
  std::vector<CoocInfo> records;
};

class CooccurrenceCollector;
class Vocab;
class CooccurrenceStatisticsHolder;
class CooccurrenceBatch;
class ResultingBufferOfCooccurrences;

class Vocab {
  friend class CooccurrenceCollector;
  friend class ResultingBufferOfCooccurrences;
  friend class CollectionParser;
 public:
  struct TokenModality {
    TokenModality() { }
    TokenModality(const std::string token_str, const std::string modality) :
                  token_str(token_str), modality(modality) { }
    std::string token_str;
    std::string modality;
  };

 private:
  Vocab();
  explicit Vocab(const std::string& path_to_vocab);
  std::string MakeKey(const std::string& token_str, const std::string& modality) const;
  int FindTokenId(const std::string& token_str, const std::string& modality) const;
  TokenModality FindTokenStr(const int token_id) const;

  std::unordered_map<std::string, int> token_map_;  // token|modality -> token_id
  std::unordered_map<int, TokenModality> inverse_token_map_;  // token_id -> (token, modality)
};

class CooccurrenceCollector {
  friend class CollectionParser;
 public:
  explicit CooccurrenceCollector(const CollectionParserConfig& config);
  unsigned VocabSize() const;
  void ReadVowpalWabbit();
  std::vector<std::string> ReadPortionOfDocuments(std::shared_ptr<std::mutex> read_lock,
                                                  std::shared_ptr<std::ifstream> vowpal_wabbit_doc_ptr);
  unsigned CooccurrenceBatchesQuantity() const;
  void ReadAndMergeCooccurrenceBatches();

 private:
  void CreateAndSetTargetFolder();
  std::string CreateFileInBatchDir() const;
  void UploadOnDisk(const CooccurrenceStatisticsHolder& cooc_stat_holder);
  void FirstStageOfMerging();
  void SecondStageOfMerging(ResultingBufferOfCooccurrences* res,
                            std::vector<std::shared_ptr<CooccurrenceBatch>>* intermediate_batches);
  void KWayMerge(ResultingBufferOfCooccurrences* res, const int mode,
                 std::vector<std::shared_ptr<CooccurrenceBatch>>* vector_of_batches_ptr,
                 std::shared_ptr<CooccurrenceBatch> out_batch,
                 std::shared_ptr<std::mutex> open_close_file_mutex_ptr);
  CooccurrenceBatch* CreateNewCooccurrenceBatch() const;
  void OpenBatchInputFile(std::shared_ptr<CooccurrenceBatch> batch);
  void OpenBatchOutputFile(std::shared_ptr<CooccurrenceBatch> batch);
  bool IsOpenBatchInputFile(const CooccurrenceBatch& batch) const;
  void CloseBatchInputFile(std::shared_ptr<CooccurrenceBatch> batch);
  void CloseBatchOutputFile(std::shared_ptr<CooccurrenceBatch> batch);

  Vocab vocab_;  // Holds mapping tokens to their indices
  std::vector<unsigned> num_of_documents_token_occurred_in_;  // index is token_id
  std::vector<std::shared_ptr<CooccurrenceBatch>> vector_of_batches_;
  int open_files_counter_;
  int64_t total_num_of_pairs_;
  unsigned total_num_of_documents_;
  CooccurrenceCollectorConfig config_;
};

class CooccurrenceStatisticsHolder {
  friend class CooccurrenceCollector;
 public:
  struct FirstToken;
  struct SecondTokenAndCooccurrence;
  void SavePairOfTokens(const int first_token_id, const int second_token_id,
                        const unsigned doc_id, const double weight = 1);
  bool Empty();

 private:
  // Here's two-level structure storage_
  // Vector was chosen because it has a very low memory overhead
  std::map<int, FirstToken> storage_;
};

struct CooccurrenceStatisticsHolder::FirstToken {
  std::map<int, SecondTokenAndCooccurrence> second_token_reference;
};

struct CooccurrenceStatisticsHolder::SecondTokenAndCooccurrence {
  explicit SecondTokenAndCooccurrence(const unsigned doc_id, const int64_t cooc_tf = 1) :
                                      last_doc_id(doc_id), cooc_tf(cooc_tf), cooc_df(1) { }
  // When a new pair comes, this field is checked and if current doc_id isn't
  // equal to previous cooc_df should be incremented
  unsigned last_doc_id;  // id of the last document where the pair occurred
  int64_t cooc_tf;
  unsigned cooc_df;
};

// Cooccurrence Batch is an intermidiate buffer between other data in RAM and
// a spesific file stored on disc. This buffer holds only one cell at a time.
// Also it's a wrapper around a ifstream and ofstream of an external file
class CooccurrenceBatch: private boost::noncopyable {
  friend class CooccurrenceCollector;
  friend class ResultingBufferOfCooccurrences;
 public:
  struct CoocBatchComparator;
  void FormNewCell(const std::map<int, CooccurrenceStatisticsHolder::FirstToken>::const_iterator& cooc_stat_node);
  void WriteCell();
  bool ReadCellHeader();
  void ReadRecords();
  bool ReadCell();  // Initiates reading of a cell from a file (e.g. call of ReadCellHeader() and ReadRecords())

 private:
  explicit CooccurrenceBatch(const std::string& path_to_batches);

  Cell cell_;
  std::ifstream in_batch_;
  std::ofstream out_batch_;
  std::string filename_;
  int64_t in_batch_offset_;
};

struct CooccurrenceBatch::CoocBatchComparator {
  bool operator()(const std::shared_ptr<CooccurrenceBatch>& left,
                  const std::shared_ptr<CooccurrenceBatch>& right) const {
    return left->cell_.first_token_id > right->cell_.first_token_id;
  }
};

class ResultingBufferOfCooccurrences {
  friend class CooccurrenceCollector;
 public:
  void CalculatePpmi();

 private:
  ResultingBufferOfCooccurrences(const Vocab& vocab,
                                 const std::vector<unsigned>& num_of_documents_token_occurred_in_,
                                 const CooccurrenceCollectorConfig& config);
  void CheckInputFile(const std::ifstream& file, const std::string& filename);
  void CheckOutputFile(const std::ofstream& file, const std::string& filename);
  void MergeWithExistingCell(const CooccurrenceBatch& batch);
  void CalculateTFStatistics();
  void WriteCoocFromCell(const std::string mode, const unsigned cooc_min);  // Output file formats are defined here
  int64_t GetCoocFromCell(const std::string& mode, const unsigned record_pos) const;
  void CalculateAndWritePpmi(const std::string mode, const long double n);
  double GetTokenFreq(const std::string& mode, const int token_id) const;

  const Vocab& vocab_;  // Holds mapping tokens to their indices
  const std::vector<unsigned>& num_of_documents_token_occurred_in_;
  std::vector<int64_t> num_of_pairs_token_occurred_in_;
  int open_files_in_buf_;
  std::ifstream cooc_tf_dict_in_;
  std::ofstream cooc_tf_dict_out_;
  std::ifstream cooc_df_dict_in_;
  std::ofstream cooc_df_dict_out_;
  std::ofstream ppmi_tf_dict_;
  std::ofstream ppmi_df_dict_;
  Cell cell_;
  CooccurrenceCollectorConfig config_;
};

}  // namespace core
}  // namespace artm
// vim: set ts=2 sw=2:
