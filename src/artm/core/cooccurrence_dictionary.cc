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

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"

using namespace std;
using namespace boost::filesystem;

// ToDo: write removal of files, creation of dir, recording batches in dir
// ToDo: replace native ptr
enum {
  MAX_NUM_OF_BATCHES = 1000,
  ITEMS_PER_BATCH = 10000,
  BATCH_STARTING_SIZE = 65536,
  CELL_STARTING_SIZE = 4096,
  MAX_NAME_LEN = 30,
  OUTPUT_BUF_SIZE = 65536,
};

static int batch_num = 0;
static int max_needed_size = 0;

typedef struct cooccurrence_info {
  int cooc_value, doc_quan, prev_doc_id;
} cooccurrence_info;

inline void FetchVocab(const char *path_to_vocab, std::unordered_map<std::string, int> &dictionary) {
  // This func reads words from vocab, sets them unique id and collects pair
  // in dictionary
  std::filebuf fb;
  if (!fb.open(path_to_vocab, std::ios::in)) {
    fprintf(stderr, "Failed to open vocab\n");
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

inline void FormName(int num, std::string &name) {
  char num_str[MAX_NAME_LEN] = "";
  sprintf(num_str,"%d", num);
  name = string("Co-occurrenceBatch");
  std::string str2 = string(num_str);
  std::string str3 = string(".bin");
  name += str2 + str3;
}

inline void DestroyAllBatches(int batch_num) {
  for (int i = 0; i < batch_num; ++i) {
    ;
  }
}

inline void UploadBatchOnDisk(std::map<int, std::map<int, cooccurrence_info>> &cooc) {
  // This function creates in a special directory a binary file wth all
  // content from the map (batch)
  // Every node of the map is stored in a cell as seqence of ints (4 bytes).
  // There are two ints and sequence of triples of ints.
  // First elem of a cell is number of triples.
  // Second element is token_id of the first token.
  // Then go triples, every triple consist of token id of the second token,
  // cooc_value and quantity of documents in which the folowing pair of
  // tokens occurred together.

  std::string name;
  FormName(batch_num, name);
  if (batch_num > MAX_NUM_OF_BATCHES) {
    fprintf(stderr, "Too many batches, maximal number of batches = %d\n",
            MAX_NUM_OF_BATCHES);
    DestroyAllBatches(batch_num);
    exit(1);
  }
  batch_num++;
  const char *cname = name.c_str();
  // ToDo: create a folder for them
  FILE *out = fopen(cname, "wb");
  if (!out) {
    fprintf(stderr, "Failed to create a file in a working directory\n");
    DestroyAllBatches(batch_num);
    exit(1);
  }
  int arr_size = BATCH_STARTING_SIZE;
  int *arr = (int *) malloc(arr_size * sizeof *arr);
  int end_of_arr = 0;

  for (auto iter = cooc.begin(); iter != cooc.end(); ++iter) {
    int needed_size = (iter->second).size() * 3;
    while (needed_size + 2 > arr_size - end_of_arr) {
      arr_size <<= 1;
      arr = (int *) realloc(arr, arr_size * sizeof *arr);
      assert(arr);
    }
    // value max_needed_size will be useful later
    // ToDo: optimize copy in arr
    if (needed_size > max_needed_size)
      max_needed_size = needed_size;
    arr[end_of_arr++] = needed_size / 3; // number of triples
    arr[end_of_arr++] = iter->first;
    for (auto iter2 = (iter->second).begin(); iter2 != (iter->second).end(); ++iter2) {
      arr[end_of_arr++] = iter2->first;
      arr[end_of_arr++] = iter2->second.cooc_value;
      arr[end_of_arr++] = iter2->second.doc_quan;
    }
  }
  fwrite(arr, sizeof *arr, end_of_arr, out);
  fclose(out);
  free(arr);
}

inline void form_new_triple(cooccurrence_info &tmp_triple, int doc_id) {
  tmp_triple.cooc_value = tmp_triple.doc_quan = 1;
  tmp_triple.prev_doc_id = doc_id;
}

inline void add_in_cooc_map(int first_token_id, int second_token_id,
        int doc_num, std::map<int, std::map<int, cooccurrence_info>> &cooc_map) {
  cooccurrence_info tmp_triple;
  form_new_triple(tmp_triple, doc_num);
  std::map<int, cooccurrence_info> tmp_map;
  tmp_map.insert( std::pair<int, cooccurrence_info> (second_token_id, tmp_triple));
  cooc_map.insert(std::pair<int, std::map<int, cooccurrence_info>> (first_token_id, tmp_map));
}

inline void modify_cooc_map_node(int second_token_id, int doc_num,
        const std::map<int, std::map<int, cooccurrence_info>>::iterator &map_record) {
  std::map<int, cooccurrence_info> *map_ptr = &map_record->second;
  auto iter = map_ptr->find(second_token_id);
  if (iter == map_ptr->end()) {
    cooccurrence_info tmp_triple;
    form_new_triple(tmp_triple, doc_num);
    map_ptr->insert(std::pair<int, cooccurrence_info> (second_token_id, tmp_triple));
  } else {
    iter->second.cooc_value++;
    if (iter->second.prev_doc_id != doc_num) {
      iter->second.prev_doc_id = doc_num;
      iter->second.doc_quan++;
    }
  }
}

inline void ReadVowpalWabbitDoc(const char *path_to_wv, const
  std::unordered_map<std::string, int> &dictionary, const int window_width) {
  // This func works as follows:
  // 1. Acquire lock for reading from vowpal wabbit file
  // 2. Read a portion (items_per_batch) of documents from file and save it
  // in a local buffer (vetor<string>)
  // 3. Release the lock
  // 4. Cut every document into words, search for them in dictionary and for
  // valid calculate co-occurrence, number of documents where these words
  // were found close enough (in a window with width window_width) and save
  // all in map
  // 5. If map isn't empty create a batch and dump all information on
  // external storage
  // Repeat 1-5 for all portions (can work in parallel for different portions)

  std::filebuf fb;
  if (!fb.open(path_to_wv, std::ios::in)) {
    fprintf(stderr, "Failed to open vowpal wabbit file\n");
    exit(1);
  }
  std::istream VowpalWabbitDoc(&fb);
  std::mutex read_lock, write_lock;

  auto func = [&dictionary, &VowpalWabbitDoc, &read_lock, &write_lock,
       &window_width]() {
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

      // The key of external map is first_token_id
      // The key of internal map is token_id of token which occured
      // together with first token
      // cooccurrence_info is a triple of ints
      // The 1nd value is quantity of co-occurences in the portion
      // The 2rd is quantity of documents (doc_quan) in which the following
      // pair of tokens occurred together
      // The 3th is number of the last document where the pair occurred
      // The 3th value is needed in calculation of doc_quan - to know, from
      // which document a previous the same pair has come
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
              add_in_cooc_map(first_token_id, second_token_id, doc_id, cooc_map);
            else
              modify_cooc_map_node(second_token_id, doc_id, map_record);
            if (swap_flag)
              first_token_id = second_token_id;
          }
        }
      }

      {
        std::lock_guard<std::mutex> guard(write_lock);
        if (!cooc_map.empty())
          UploadBatchOnDisk(cooc_map);
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

enum {
  FIRST_TOKEN_ID = 0,
  NUM_OF_TRIPLES = 1,
  LIST_OF_TRIPLES = 2,
};

typedef struct token_and_cooc_info {
  int token_id, cooc_value, doc_quan;
} token_and_cooc_info;

typedef struct batch_struct {
  FILE *file;
  int cell_size; // size of useful memory
  int first_token_id;
} batch_struct;

struct BatchHeapComparator {
  int operator() (const batch_struct &left, const batch_struct &right) const {
    return left.first_token_id > right.first_token_id;
  }
};

inline FILE *CreateResFile() {
  return fopen("Co-occurrenceDictionary.txt", "w");
}

inline int ReadCellHeader(std::vector<batch_struct> &arr_batch, int ind) {
  const int items_to_read = 2;
  int tmp_buf[items_to_read];
  int read_items = fread(tmp_buf, sizeof(int), items_to_read, arr_batch[ind].file);
  arr_batch[ind].cell_size = tmp_buf[0];
  arr_batch[ind].first_token_id = tmp_buf[1];
  return read_items == items_to_read;
}

inline void ArrBatchInitialization(std::vector<batch_struct> &arr_batch) {
  for (int i = 0; i < (int64_t) arr_batch.size(); ++i) {
    std::string name;
    FormName(i, name);
    arr_batch[i].file = fopen(name.c_str(), "rb");
    if (!arr_batch[i].file) {
      fprintf(stderr, "Failed to access to batch number %d\n", i);
      DestroyAllBatches(batch_num);
      exit(1);
    }
    ReadCellHeader(arr_batch, i);
  }
}

inline void FormListFromTriples(token_and_cooc_info *addr, int triples_num,
        std::list<token_and_cooc_info> &res) {
  for (int i = 0; i < triples_num; ++i)
    res.push_back(addr[i]);
}

inline void CheckCooccurrenceFreq(std::vector<std::tuple<int, int,
        std::list<token_and_cooc_info>>> &result, const int min_cooc_value) {
  for (auto iter = std::get<LIST_OF_TRIPLES>(result[0]).begin();
           iter != std::get<LIST_OF_TRIPLES>(result[0]).end(); ) {
    if (iter->cooc_value < min_cooc_value) {
      std::get<LIST_OF_TRIPLES>(result[0]).erase(iter++);
    } else
      iter++;
  }
  if (std::get<LIST_OF_TRIPLES>(result[0]).empty())
    result.pop_back();
}

// ToDo: Implement with sorted in ascending order forward list
inline void MergeListsWithAddition(token_and_cooc_info *addr, int triples_num,
        std::list<token_and_cooc_info> &tmp_list) {
  auto iter2 = tmp_list.begin();
  int iter1 = 0;
  for (; iter1 < triples_num && iter2 != tmp_list.end(); ) {
    if (addr[iter1].token_id == iter2->token_id) {
      iter2->cooc_value += addr[iter1].cooc_value;
      iter2->doc_quan += addr[iter1].doc_quan;
      iter1++;
      iter2++;
    } else if (addr[iter1].token_id > iter2->token_id)
      iter2++;
    else
      tmp_list.insert(iter2, addr[iter1++]);
  }
  for (; iter1 < triples_num; ) {
    tmp_list.push_back(addr[iter1++]);
  }
}

inline void WriteInResFile(std::pair<int, int *> &output_buf, FILE *res_file) {
  //fwrite(output_buf.third, sizeof(int), output_buf.first, res_file);
  for (int i = 0; i < output_buf.first; i += 3)
    fprintf(res_file, "%d %d %d\r\n", output_buf.second[i],
              output_buf.second[i + 1], output_buf.second[i + 2]);
  output_buf.first = 0;
}

inline void DumpResRecordInBuf(std::pair<int, int *> &output_buf,
        std::vector<std::tuple<int, int, std::list<token_and_cooc_info>>> &res,
        FILE *res_file) {
  if (std::get<NUM_OF_TRIPLES>(res[0]) * 3 > OUTPUT_BUF_SIZE - output_buf.first)
    WriteInResFile(output_buf, res_file);
  for (auto iter = std::get<LIST_OF_TRIPLES>(res[0]).begin();
           iter != std::get<LIST_OF_TRIPLES>(res[0]).end(); ++iter) {
    output_buf.second[output_buf.first++] = std::get<FIRST_TOKEN_ID>(res[0]);
    output_buf.second[output_buf.first++] = iter->token_id;
    output_buf.second[output_buf.first++] = iter->cooc_value;
  }
}

inline void ReadAndMergeBatches(const int min_cooc_value) {
  std::vector<batch_struct> arr_batch (batch_num);
  ArrBatchInitialization(arr_batch);
  std::make_heap(arr_batch.begin(), arr_batch.end(), BatchHeapComparator());
  FILE *res_file = CreateResFile();
  if (!res_file) {
    fprintf(stderr, "Failed to create a file in a working directory\n");
    DestroyAllBatches(batch_num);
    exit(1);
  }

  // This vector won't hold more than 1 element, because another element
  // means another first_token_id => all the data linked with current
  // first_token_id can be filtered and be ready to be written in
  // resulting file.
  // The first element in the tuple is first_token_id.
  // The second is number of triples that are kept in the third element
  std::vector<std::tuple<int, int, std::list<token_and_cooc_info>>> result;

  // From output_buf data will be loaded in resulting file
  // The first elem here is number of busy ints in buf.
  // The second is pointer to buf
  std::pair<int, int *> output_buf;
  output_buf.second = (int *) malloc(OUTPUT_BUF_SIZE * sizeof *output_buf.second);

  // Data will be loaded in tmp_buf from batches and then into result vector
  int *tmp_buf = (int *) malloc(max_needed_size * sizeof(int));

  // Standard k-way merge as external sort
  while (!arr_batch.empty()) {
    // It's guaranteed that batches aren't empty (look ParseVowpalWabbitDoc func)
    std::list<token_and_cooc_info> tmp_list;
    fread(tmp_buf, sizeof(token_and_cooc_info), arr_batch[0].cell_size, arr_batch[0].file);
    if (result.empty()) {
      FormListFromTriples((token_and_cooc_info *) tmp_buf, arr_batch[0].cell_size, tmp_list);
      result.push_back(std::tuple<int, int, std::list<token_and_cooc_info>>
            (arr_batch[0].first_token_id, arr_batch[0].cell_size, tmp_list));
    } else if (std::get<FIRST_TOKEN_ID>(result[0]) == arr_batch[0].first_token_id) {
      MergeListsWithAddition((token_and_cooc_info *) tmp_buf, arr_batch[0].cell_size,
              std::get<LIST_OF_TRIPLES>(result[0]));
      std::get<NUM_OF_TRIPLES>(result[0]) += arr_batch[0].cell_size;
    } else {
      CheckCooccurrenceFreq(result, min_cooc_value);
      if (!result.empty()) {
        DumpResRecordInBuf(output_buf, result, res_file);
        result.pop_back();
      }
      FormListFromTriples((token_and_cooc_info *) tmp_buf, arr_batch[0].cell_size, tmp_list);
      result.push_back(std::tuple<int, int, std::list<token_and_cooc_info>>
            (arr_batch[0].first_token_id, arr_batch[0].cell_size, tmp_list));
    }

    std::pop_heap(arr_batch.begin(), arr_batch.end(), BatchHeapComparator());
    if (ReadCellHeader(arr_batch, arr_batch.size() - 1))
      std::push_heap(arr_batch.begin(), arr_batch.end(), BatchHeapComparator());
    else {
      fclose(arr_batch[arr_batch.size() - 1].file);
      arr_batch.pop_back();
    }
  }
  CheckCooccurrenceFreq(result, min_cooc_value);
  if (!result.empty())
    DumpResRecordInBuf(output_buf, result, res_file);
  if (output_buf.first)
    WriteInResFile(output_buf, res_file);
  free(tmp_buf);
  free(output_buf.second);
  fclose(res_file);
}

// command line interface:
// ./main path/to/VowpalWabbitDoc path/to/vocab window_width min_cooc_value
int main(int argc, char **argv) {
  const int window_width   = atoi(argv[3]);
  const int min_cooc_value = atoi(argv[4]);

  // This function works as follows:
  // 1. Get content from a vocab file and put it in dictionary
  // 2. Read Vowpall Wabbit file by portions, calculate co-occurrences for
  // every portion and save it (batch) on external storage
  // 3. Read from external storage all the batches piece by piece and create
  // resulting file with all co-occurrences

  // If no co-occurrence found or it's low than min_cooc_value file isn't
  // created
  std::unordered_map<string, int> dictionary;
  FetchVocab(argv[2], dictionary);
  if (dictionary.size() > 1) {
    ReadVowpalWabbitDoc(argv[1], dictionary, window_width);
    if (batch_num) {
      ReadAndMergeBatches(min_cooc_value);
      DestroyAllBatches(batch_num);
    }
  }
  return 0;
}
