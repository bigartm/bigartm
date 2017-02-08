#include <cstdio>
#include <unordered_map>
#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <map>
#include <vector>
#include <list>
#include <tuple>
#include <assert.h>
#include <future>
#include <mutex>
#include <sstream>
#include <iomanip>

#include <boost/core/noncopyable.hpp>
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"

using namespace std;
using namespace boost;

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

class Batch: private boost::noncopyable {
  struct Cell {
    int first_token_id;
    int num_of_triples;
    std::vector<Triple> records;
  };

  friend class BatchManager;
 private:
  std::ifstream in_batch;
  std::ofstream out_batch;

  Batch(int batch_num, const char filemode, const string &disk_path) {
    const int max_name_len = 30;
    char num_str[max_name_len] = "";
    sprintf(num_str,"%.3d", batch_num);
    std::string filename = string(num_str);
    filename += string(".bin");
    boost::filesystem::path full_filename = boost::filesystem::path(disk_path)
                   / boost::filesystem::path(filename);
    if (filemode == 'w') {
      out_batch = boost::filesystem::ofstream(full_filename);
      if (out_batch.bad()) {
        std::cerr << "Batch::Batch: Failed to create batch\n";
        throw 1;
      }
    } else {
      in_batch = boost::filesystem::ifstream(full_filename);
      if (in_batch.bad()) {
        std::cerr << "Batch::Batch: Failed to open existing batch\n";
        throw 1;
      }
    }
  }
 public:
  Cell cell;

  void FormNewCell(std::map<int, std::map<int, cooccurrence_info>>::iterator &map_node) {
    cell.first_token_id = map_node->first;
    cell.records.resize(0);
    for (auto iter = (map_node->second).begin(); iter != (map_node->second).end(); ++iter) {
      Triple tmp;
      tmp.cooc_value = iter->second.cooc_value;
      tmp.doc_quan   = iter->second.doc_quan;
      tmp.second_token_id = iter->first;
      cell.records.push_back(tmp);
    }
    cell.num_of_triples = cell.records.size();
  }

  int ReadCellHeader() {
    in_batch.read(reinterpret_cast<char *>(&cell), 2 * sizeof(int));
    if (!in_batch.eof())
      return true;
    else {
      cell.first_token_id = -1;
      cell.num_of_triples = 0;
      return false;
    }
  }

  void ReadRecords() {
    cell.records.resize(cell.num_of_triples);
    in_batch.read(reinterpret_cast<char *>(&cell.records[0]),
            sizeof(Triple) * cell.num_of_triples);
  }

  int ReadCell() {
    if(ReadCellHeader()) {
      ReadRecords();
      return true;
    }
    return false;
  }

  void WriteCell() {
    out_batch.write(reinterpret_cast<char *>(&cell), 2 * sizeof(int));
    out_batch.write(reinterpret_cast<char *>(&cell.records[0]),
            sizeof(Triple) * cell.num_of_triples);
  }
};

class BatchManager {
 private:
  int batch_quan;
  const int max_batch_quan = 1000;
  std::string path_to_batches;
 public:
  BatchManager() {
    batch_quan = 0;
    boost::filesystem::path dir("Co-occurrenceBatches");
    if (boost::filesystem::exists(dir)) {
      std::cerr << "Folder with name Co-occurrenceBatch already exists in the working directory, please replace it\n";
      throw 1;
    }
    if (!boost::filesystem::create_directory(dir)) {
      std::cerr << "Failed to create directory\n";
      throw 1;
    }
    //boost::filesystem::path full_dirname =
    //    boost::filesystem::path(boost::filesystem::current_path()) /
    //    boost::filesystem::path(dir);
    path_to_batches = /*full_dirname.string()*/ dir.string();
  }

  ~BatchManager() { boost::filesystem::remove_all(path_to_batches); }

  int GetBatchQuan() { return batch_quan; }

  Batch *CreateNewBatch() {
    if (batch_quan < max_batch_quan) {
      return new Batch(batch_quan++, 'w', path_to_batches);
    } else {
      std::cerr << "Too many batches, maximal number of batches = "
                << max_batch_quan << endl;
      throw 1;
    }
  }

  Batch *OpenExistingBatch(int batch_num) {
    return new Batch(batch_num, 'r', path_to_batches);
  }
};

class ResultingBuffer {
 private:
  double cooc_min_tf;
  int cooc_min_df;
  int first_token_id;
  std::vector<Triple> rec;
  std::ofstream cooc_dictionary;
  std::ofstream doc_quan_dictionary;

  void MergeWithExistingCell(const Batch *batch) {
    std::vector<Triple> new_vector;
    auto fi_iter = rec.begin();
    auto se_iter = batch->cell.records.begin();
    while (fi_iter != rec.end() && se_iter != batch->cell.records.end()) {
      if (fi_iter->second_token_id == se_iter->second_token_id) {
        Triple tmp;
        tmp.second_token_id = fi_iter->second_token_id;
        tmp.cooc_value = fi_iter->cooc_value + se_iter->cooc_value;
        tmp.doc_quan = fi_iter->doc_quan + se_iter->doc_quan;
        new_vector.push_back(tmp);
        fi_iter++, se_iter++;
      } else if (fi_iter->second_token_id < se_iter->second_token_id)
        new_vector.push_back(*(fi_iter++));
      else
        new_vector.push_back(*(se_iter++));
    }
    std::copy(fi_iter, rec.end(), std::back_inserter(new_vector));
    std::copy(se_iter, batch->cell.records.end(), std::back_inserter(new_vector));
    rec = new_vector;
  }
  // Note: here is cast to int and comparison of doubles
  void PopPreviousContent() {
    for (int i = 0; i < (int) rec.size(); ++i) {
      if (rec[i].cooc_value >= cooc_min_tf)
        cooc_dictionary << first_token_id << " " << rec[i].second_token_id
                 << " " << (int) rec[i].cooc_value << endl;
      if (rec[i].doc_quan   >= cooc_min_df)
        doc_quan_dictionary << first_token_id << " " << rec[i].second_token_id
                 << " " << rec[i].doc_quan << endl;
    }
  }

  void AddNewCellInBuffer(const Batch *batch) {
    first_token_id = batch->cell.first_token_id;
    rec = batch->cell.records;
  }
 public:
  ResultingBuffer(const double min_tf, const int min_df) {
    // No need to check if buffer's empty. At first usage new data will need
    // to be pushed while previous popped, but previous data doesn't exist
    // (see AddInBuffer and PopPreviousContent methods)
    cooc_min_tf = min_tf;
    cooc_min_df = min_df;
    first_token_id = -1;
    cooc_dictionary.open("Co-occurrenceDictionary.txt", std::ios::out);
    doc_quan_dictionary.open("DocQuanDictionary.txt",   std::ios::out);
    if (cooc_dictionary.bad() || doc_quan_dictionary.bad()) {
      std::cerr << "Failed to create a file in the working directory\n";
      throw 1;
    }
  }

  ~ResultingBuffer() { PopPreviousContent(); }

  void AddInBuffer(const Batch *batch) {
    if (first_token_id == batch->cell.first_token_id) {
      MergeWithExistingCell(batch);
    } else {
      PopPreviousContent();
      AddNewCellInBuffer(batch);
    }
  }
};

class CooccurrenceDictionary {
 private:
  int window_width;
  int cooc_min_df;
  double cooc_min_tf;
  std::string path_to_vocab;
  std::string path_to_vw;
  BatchManager batch_manager;
  std::unordered_map<string, int> dictionary;

  void FetchVocab(const std::string &path_to_vocab,
        std::unordered_map<std::string, int> &dictionary);

  void UploadBatchOnDisk(BatchManager &batch_manager,
        std::map<int, std::map<int, cooccurrence_info>> &cooc);

  void AddInCoocMap(int first_token_id, int second_token_id, int doc_id,
        std::map<int, std::map<int, cooccurrence_info>> &cooc_map);

  void ModifyCoocMapNode(int second_token_id, int doc_id,
          std::map<int, cooccurrence_info> &map_node);

  void ReadVowpalWabbit(const std::string &path_to_vw, const int window_width,
        const std::unordered_map<std::string, int> &dictionary,
        BatchManager &batch_manager);

  void ReadAndMergeBatches(const double cooc_min_tf, const int cooc_min_df,
          BatchManager &batch_manager);
 public:
  CooccurrenceDictionary(const char *vw, const char *vocab,
          const char *wind_width, const char *min_tf, const char *min_df) {
    // This function works as follows:
    // 1. Get content from a vocab file and put it in dictionary
    // 2. Read Vowpal Wabbit file by portions, calculate co-occurrences for
    // every portion and save it (batch) on external storage
    // 3. Read from external storage all the batches piece by piece and create
    // resulting file with all co-occurrences

    // If no co-occurrence found or it's low than min_cooc_value file isn't
    // created
    path_to_vw = vw;
    path_to_vocab = vocab;
    window_width = atoi(wind_width);
    cooc_min_tf = atof(min_tf);
    cooc_min_df = atoi(min_df);
    FetchVocab(path_to_vocab, dictionary);
    if (dictionary.size() > 1) {
      ReadVowpalWabbit(path_to_vw, window_width, dictionary, batch_manager);
      if (batch_manager.GetBatchQuan())
        ReadAndMergeBatches(cooc_min_tf, cooc_min_df, batch_manager);
    }
  }
};

void CooccurrenceDictionary::FetchVocab(const std::string &path_to_vocab,
        std::unordered_map<std::string, int> &dictionary) {
  // This func reads words from vocab, sets them unique id and collects pair
  // in dictionary
  std::filebuf fb;
  if (!fb.open(path_to_vocab, std::ios::in)) {
    std::cerr << "Failed to open vocab\n";
    throw 1;
  }
  std::istream vocab(&fb);
  int last_token_id = 1;
  std::string str;

  while (true) {
    getline(vocab, str);
    if (vocab.eof())
      break;
    boost::algorithm::trim(str);
    if (!str.empty())
      dictionary.insert(std::make_pair(str, last_token_id++));
  }
}

void CooccurrenceDictionary::UploadBatchOnDisk(BatchManager &batch_manager,
        std::map<int, std::map<int, cooccurrence_info>> &cooc) {
  Batch *batch = batch_manager.CreateNewBatch();
  for (auto iter = cooc.begin(); iter != cooc.end(); ++iter) {
    batch->FormNewCell(iter);
    batch->WriteCell();
  }
  delete batch;
}

void CooccurrenceDictionary::AddInCoocMap(int first_token_id,
        int second_token_id, int doc_id,
        std::map<int, std::map<int, cooccurrence_info>> &cooc_map) {
  cooccurrence_info tmp;
  tmp.doc_quan = tmp.cooc_value = 1;
  tmp.prev_doc_id = doc_id;
  std::map<int, cooccurrence_info> tmp_map;
  tmp_map.insert( std::pair<int, cooccurrence_info> (second_token_id, tmp));
  cooc_map.insert(std::pair<int, std::map<int, cooccurrence_info>> (first_token_id, tmp_map));
}

void CooccurrenceDictionary::ModifyCoocMapNode(int second_token_id,
        int doc_id, std::map<int, cooccurrence_info> &map_node) {
  auto iter = map_node.find(second_token_id);
  if (iter == map_node.end()) {
    cooccurrence_info tmp;
    tmp.doc_quan = tmp.cooc_value = 1;
    tmp.prev_doc_id = doc_id;
    map_node.insert(std::pair<int, cooccurrence_info> (second_token_id, tmp));
  } else {
    iter->second.cooc_value++;
    if (iter->second.prev_doc_id != doc_id) {
      iter->second.prev_doc_id = doc_id;
      iter->second.doc_quan += 1;
    }
  }
}

void CooccurrenceDictionary::ReadVowpalWabbit(const std::string &path_to_vw,
        const int window_width,
        const std::unordered_map<std::string, int> &dictionary,
        BatchManager &batch_manager) {
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

  std::filebuf fb;
  if (!fb.open(path_to_vw, std::ios::in)) {
    std::cerr << "Failed to open vocab\n";
    throw 1;
  }
  std::istream VowpalWabbitDoc(&fb);
  std::mutex read_lock, write_lock;

  auto func = [&]() {
    while (true) {
      std::vector<std::string> portion;

      {
        std::lock_guard<std::mutex> guard(read_lock);
        if (VowpalWabbitDoc.eof())
          return;

        std::string str;
        while (portion.size() < ITEMS_PER_BATCH) {
          getline(VowpalWabbitDoc, str);
          if (VowpalWabbitDoc.eof())
            break;
          portion.push_back(str);
        }
      }

      if (!portion.size())
        continue;

      // First elem in external map is first_token_id, in internal it's
      // second_token_id
      std::map<int, std::map<int, cooccurrence_info>> cooc_map;

      for (int doc_id = 0; doc_id < (int64_t) portion.size(); ++doc_id) {
        std::vector<std::string> doc;
        boost::split(doc, portion[doc_id], boost::is_any_of(" \t\r"));
        if (doc.size() <= 1)
          continue;

        for (int j = 1; j < (int64_t) doc.size() - 1; ++j) {
          auto first_token = dictionary.find(doc[j]);
          if (first_token == dictionary.end())
            continue;
          int first_token_id = first_token->second;

          for (int k = 1; k < window_width && j + k < (int64_t) doc.size(); ++k) {
            auto second_token = dictionary.find(doc[j + k]);
            if (second_token == dictionary.end())
              continue;
            int second_token_id = second_token->second;
            if (first_token_id == second_token_id)
              continue;
            int swap_flag = 0;

            if (first_token_id >  second_token_id) {
              swap_flag = 1;
              std::swap(first_token_id, second_token_id);
            }
            auto map_record = cooc_map.find(first_token_id);
            if (map_record == cooc_map.end())
              AddInCoocMap(first_token_id, second_token_id, doc_id, cooc_map);
            else
              ModifyCoocMapNode(second_token_id, doc_id, map_record->second);
            if (swap_flag)
              first_token_id = second_token_id;
          }
        }
      }

      {
        std::lock_guard<std::mutex> guard(write_lock);
        if (!cooc_map.empty())
          UploadBatchOnDisk(batch_manager, cooc_map);
      }
    }
  };
  int num_of_threads = std::thread::hardware_concurrency();
  if (!num_of_threads)
    num_of_threads = 1;
  std::vector<std::shared_future<void>> tasks;
  for (int i = 0; i < num_of_threads; ++i)
    tasks.push_back(std::move(std::async(std::launch::async, func)));
  for (int i = 0; i < num_of_threads; ++i)
    tasks[i].get();
}

void CooccurrenceDictionary::ReadAndMergeBatches(const double cooc_min_tf,
        const int cooc_min_df, BatchManager &batch_manager) {
  auto CompareBatches = [](const Batch *left, const Batch *right) {
    return left->cell.first_token_id > right->cell.first_token_id;
  };
  std::vector<Batch *> batch_queue;
  for (int i = 0; i < batch_manager.GetBatchQuan(); ++i) {
    batch_queue.push_back(batch_manager.OpenExistingBatch(i));
    batch_queue[i]->ReadCell();
  }
  std::make_heap(batch_queue.begin(), batch_queue.end(), CompareBatches);

  // This buffer won't hold more than 1 element, because another element
  // means another first_token_id => all the data linked with current
  // first_token_id can be filtered and be ready to be written in
  // resulting file.
  ResultingBuffer res(cooc_min_tf, cooc_min_df);

  // Standard k-way merge as external sort
  while (!batch_queue.empty()) {
    // It's guaranteed that batches aren't empty (look ParseVowpalWabbit func)
    res.AddInBuffer(batch_queue[0]);
    std::pop_heap(batch_queue.begin(), batch_queue.end(), CompareBatches);
    if (batch_queue.back()->ReadCell()) {
      std::push_heap(batch_queue.begin(), batch_queue.end(), CompareBatches);
    } else {
      delete batch_queue.back();
      batch_queue.pop_back();
    }
  }
}

// command line interface:
// ./main path/to/VowpalWabbitDoc path/to/vocab window_width cooc_min_tf
// cooc_min_df
int main(int argc, char **argv) {
  CooccurrenceDictionary(argv[1], argv[2], argv[3], argv[4], argv[5]);
  return 0;
}
