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

#include "boost/algorithm/string.hpp"
//#include "boost/filesystem.hpp"

// ToDo: in all places we throw exception, we need to call destructor of
// BatchManager
// ToDo: remove frptinf(stderr) and exit(1)
// ToDo: write removal of files, creation of dir, recording batches in dir
// (boost)
// ToDo: optimize io operations with files
// ToDo: replace FILE *
// ToDo: make fclose of FILE * in class, make FILE *file private
using namespace std;
//using namespace boost::filesystem;

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

class Batch {
  struct Cell {
    int first_token_id;
    int num_of_triples;
    std::vector<Triple> records;
  };
  friend class BatchManager;
 public:
  FILE *file = nullptr;
 private:
  Batch(int batch_num, const char file_mode[]) {
    const int max_name_len = 30;
    std::string name = string("Co-occurrenceBatch");
    char num_str[max_name_len] = "";
    sprintf(num_str,"%d", batch_num);
    std::string str2 = string(num_str);
    std::string str3 = string(".bin");
    name += str2 + str3;
    const char *cname = name.c_str();
    file = fopen(cname, file_mode);
    if (!file) {
      std::stringstream ss;
      if (!strcmp(file_mode, "wb"))
        ss << "Failed to create a file in the working directory\n";
      else
        ss << "Failed to open existing file in the working directory\n";
      //BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
      fprintf(stderr, "line 71");
      exit(1);
    }
  }
 public:
  Cell cell;

  ~Batch() {
    //fclose(file);
  }
  // ToDo: think how to optimize copy in cell
  void FormNewCell(std::map<int, std::map<int, Pair>>::iterator &map_node) {
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
    if (fread(&cell, sizeof(int), 2, file) == 2)
      return true;
    else {
      cell.first_token_id = -1;
      cell.num_of_triples = 0;
      return false;
    }
  }
  void ReadRecords() {
    cell.records.resize(cell.num_of_triples);
    fread(&cell.records[0], sizeof(Triple), cell.num_of_triples, file);
  }
  int ReadCell() {
    if(ReadCellHeader()) {
      ReadRecords();
      return true;
    }
    return false;
  }
  void WriteCell() {
    fwrite(&cell.first_token_id, sizeof(cell.first_token_id), 1, file);
    fwrite(&cell.num_of_triples, sizeof(cell.num_of_triples), 1, file);
    fwrite(&cell.records[0], sizeof(Triple), cell.num_of_triples, file);
  }
};

class BatchManager {
 private:
  int batch_quan = 0;
  const int max_batch_quan = 1000;
 public:
  BatchManager() {
    // create a folder;
  }
  ~BatchManager() {
    // delete whole folder;
  }
  int GetBatchQuan() { return batch_quan; }
  Batch CreateNewBatch() {
    if (batch_quan < max_batch_quan) {
      return Batch(batch_quan++, "wb");
    } else {
      std::stringstream ss;
      ss << "Too many batches, maximal number of batches = "
         << max_batch_quan << endl;
      // delete whole folder
      //BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
      fprintf(stderr, "line 143");
      exit(1);
    }
  }
  Batch OpenExistingBatch(int batch_num) { return Batch(batch_num, "rb"); }
};

class ResultingBuffer {
 private:
  double min_cooc_value;
  int first_token_id;
  std::vector<Triple> rec;
  std::ofstream cooc_dictionary;
  std::ofstream doc_quan_dictionary;

  void MergeWithExistingCell(const Batch &batch) {
    std::vector<Triple> new_vector;
    auto fi_iter = rec.begin();
    auto se_iter = batch.cell.records.begin();
    while (fi_iter != rec.end() && se_iter != batch.cell.records.end()) {
      if (fi_iter->second_token_id == se_iter->second_token_id) {
        Triple tmp;
        tmp.second_token_id = fi_iter->second_token_id;
        tmp.cooc_value = fi_iter->cooc_value + se_iter->cooc_value;
        tmp.doc_quan = fi_iter->doc_quan + se_iter->doc_quan;
        new_vector.push_back(tmp);
        fi_iter++, se_iter++;
      } else if (fi_iter->second_token_id < se_iter->second_token_id)
        new_vector.push_back(*fi_iter);
      else
        new_vector.push_back(*se_iter);
    }
    // ToDo: memcpy can be used here
    while (fi_iter != rec.end())
      new_vector.push_back(*fi_iter);
    while (se_iter != batch.cell.records.end())
      new_vector.push_back(*se_iter);
    rec = new_vector;
  }
  // Note: here is cast to int
  void PopPreviousContent() {
    for (int i = 0; i < (int) rec.size(); ++i)
      if (rec[i].cooc_value > min_cooc_value) {
        cooc_dictionary << first_token_id << " "
                     << rec[i].second_token_id << " "
                     << (int) rec[i].cooc_value << endl;
        doc_quan_dictionary << first_token_id << " "
                << rec[i].second_token_id << " " << rec[i].doc_quan << endl;
      }
  }
  void AddNewCellInBuffer(const Batch &batch) {
    first_token_id = batch.cell.first_token_id;
    rec = batch.cell.records;
  }
 public:
  ResultingBuffer(const double min_cooc_val) {
    // No need to check if buffer's empty. At first usage new data will need
    // to be pushed while previous popped, but previous data doesn't exist
    // (see AddInBuffer and PopPreviousContent methods)
    min_cooc_value = min_cooc_val;
    first_token_id = -1;
    cooc_dictionary.open("Co-occurrenceDictionary.txt", std::ios::out);
    doc_quan_dictionary.open("DocQuanDictionary.txt",   std::ios::out);
    if (cooc_dictionary.bad() || doc_quan_dictionary.bad()) {
      std::stringstream ss;
      ss << "Failed to create a file in the working directory\n";
      //BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
      fprintf(stderr, "line 201");
      exit(1);
    }
  }
  ~ResultingBuffer() {
    PopPreviousContent();
    cooc_dictionary.close();
    doc_quan_dictionary.close();
  }
  void AddInBuffer(const Batch &batch) {
    if (first_token_id == batch.cell.first_token_id)
      MergeWithExistingCell(batch);
    else {
      PopPreviousContent();
      AddNewCellInBuffer(batch);
    }
  }
};

inline void FetchVocab(const char *path_to_vocab, std::unordered_map<std::string, int> &dictionary) {
  // This func reads words from vocab, sets them unique id and collects pair
  // in dictionary
  std::filebuf fb;
  if (!fb.open(path_to_vocab, std::ios::in)) {
    std::stringstream ss;
    ss << "Failed to open vocab\n";
    //BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    fprintf(stderr, "line 246");
    exit(1);
  }
  std::istream vocab(&fb);
  int last_token_id = 1;
  std::string str;

  while (true) {
    getline(vocab, str);
    if (vocab.eof())
      break;
    dictionary.insert(std::make_pair(str, last_token_id++));
  }
}

inline void UploadBatchOnDisk(BatchManager &batch_manager,
        std::map<int, std::map<int, Pair>> &cooc) {
  Batch batch = batch_manager.CreateNewBatch();
  for (auto iter = cooc.begin(); iter != cooc.end(); ++iter) {
    batch.FormNewCell(iter);
    batch.WriteCell();
  }
  fclose(batch.file);
}

inline void AddInCoocMap(int first_token_id, int second_token_id,
        int doc_num, std::map<int, std::map<int, Pair>> &cooc_map) {
  Pair tmp_pair;
  tmp_pair.doc_quan = tmp_pair.cooc_value = 1;
  std::map<int, Pair> tmp_map;
  tmp_map.insert( std::pair<int, Pair> (second_token_id, tmp_pair));
  cooc_map.insert(std::pair<int, std::map<int, Pair>> (first_token_id, tmp_map));
}

inline void ModifyCoocMapNode(int second_token_id, int doc_num,
        std::map<int, Pair> &map_node) {
  auto iter = map_node.find(second_token_id);
  if (iter == map_node.end()) {
    Pair tmp_pair;
    tmp_pair.doc_quan = tmp_pair.cooc_value = 1;
    map_node.insert(std::pair<int, Pair> (second_token_id, tmp_pair));
  } else
    iter->second.cooc_value++;
}

// ToDo: can be optimized
inline void MergeMaps(std::vector<std::map<int, std::map<int, Pair>>> &vector_maps) {
  for (int i = vector_maps.size() - 1; i > 0; --i)
    for (auto iter = vector_maps[i].begin(); iter != vector_maps[i].end(); ++iter) {
      auto iter2 = vector_maps[0].find(iter->first);
      if (iter2 == vector_maps[0].end()) {
        vector_maps[0].insert(*iter);
      } else
        for (auto iter3 = iter->second.begin(); iter3 != iter->second.end(); ++iter3) {
          auto iter4 = iter2->second.find(iter3->first);
          if (iter4 == iter2->second.end()) {
            iter2->second.insert(*iter3);
          } else {
            iter4->second.doc_quan++;
            iter4->second.cooc_value += iter3->second.cooc_value;
          }
        }
    }
}

inline void ReadVowpalWabbit(const char *path_to_vw, const int window_width,
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
    std::stringstream ss;
    ss << "Failed to open vocab\n";
    //BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    fprintf(stderr, "line 313");
    exit(1);
  }
  std::istream VowpalWabbitDoc(&fb);
  std::mutex read_lock, write_lock;

  auto func = [&dictionary, &VowpalWabbitDoc, &read_lock, &write_lock,
         &window_width, &batch_manager]() {
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
      // ToDo: maybe would be better to use map of sorted lists here
      std::vector<std::map<int, std::map<int, Pair>>> vector_maps(portion.size());

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
            auto map_record = vector_maps[doc_id].find(first_token_id);
            if (map_record == vector_maps[doc_id].end())
              AddInCoocMap(first_token_id, second_token_id, doc_id, vector_maps[doc_id]);
            else
              ModifyCoocMapNode(second_token_id, doc_id, map_record->second);
            if (swap_flag)
              first_token_id = second_token_id;
          }
        }
      }
      MergeMaps(vector_maps);

      {
        std::lock_guard<std::mutex> guard(write_lock);
        if (!vector_maps[0].empty())
          UploadBatchOnDisk(batch_manager, vector_maps[0]);
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

inline void ReadAndMergeBatches(const double min_cooc_value,
        BatchManager &batch_manager) {
  auto CompareBatches = [](const Batch &left, const Batch &right) {
    return left.cell.first_token_id > right.cell.first_token_id;
  };
  std::vector<Batch> batch_queue;
  for (int i = 0; i < batch_manager.GetBatchQuan(); ++i) {
    batch_queue.push_back(batch_manager.OpenExistingBatch(i));
    batch_queue[i].ReadCell();
  }
  std::make_heap(batch_queue.begin(), batch_queue.end(), CompareBatches);

  // This buffer won't hold more than 1 element, because another element
  // means another first_token_id => all the data linked with current
  // first_token_id can be filtered and be ready to be written in
  // resulting file.
  ResultingBuffer res(min_cooc_value);

  // Standard k-way merge as external sort
  while (!batch_queue.empty()) {
    // It's guaranteed that batches aren't empty (look ParseVowpalWabbit func)
    res.AddInBuffer(batch_queue[0]);
    std::pop_heap(batch_queue.begin(), batch_queue.end(), CompareBatches);
    if (batch_queue[batch_queue.size() - 1].ReadCell())
      std::push_heap(batch_queue.begin(), batch_queue.end(), CompareBatches);
    else {
      fclose(batch_queue[batch_queue.size() - 1].file);
      batch_queue.pop_back();
    }
  }
}

// command line interface:
// ./main path/to/VowpalWabbitDoc path/to/vocab window_width min_cooc_value
int main(int argc, char **argv) {
  const int window_width      = atoi(argv[3]);
  const double min_cooc_value = atof(argv[4]);
  BatchManager batch_manager;

  // This function works as follows:
  // 1. Get content from a vocab file and put it in dictionary
  // 2. Read Vowpal Wabbit file by portions, calculate co-occurrences for
  // every portion and save it (batch) on external storage
  // 3. Read from external storage all the batches piece by piece and create
  // resulting file with all co-occurrences

  // If no co-occurrence found or it's low than min_cooc_value file isn't
  // created
  std::unordered_map<string, int> dictionary;
  FetchVocab("vocab", dictionary);
  if (dictionary.size() > 1) {
    ReadVowpalWabbit("vw", window_width, dictionary, batch_manager);
    if (batch_manager.GetBatchQuan())
      ReadAndMergeBatches(min_cooc_value, batch_manager);
  }
  return 0;
}
