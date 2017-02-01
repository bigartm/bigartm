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

// ToDo: replace it with variables
enum {
  ITEMS_PER_BATCH = 10000,
  WINDOW_WIDTH = 5,
  BATCH_STARTING_SIZE = 65536,
  CELL_STARTING_SIZE = 4096,
  MAX_NAME_LEN = 30,
  OUTPUT_BUF_SIZE = 65536,
  MIN_COOC_VALUE = 1,
};

enum {
  FIRST_TOKEN_ID = 0,
  NUM_OF_TRIPLES = 1,
  LIST_OF_TRIPLES = 2
};

// ToDo: replace global variables with class
static int batch_num = 0;
static int max_needed_size = 0;

typedef struct triple_int {
  int token_id, cooc_value, doc_quan;
} triple_int;

typedef struct quad_int {
  int token_id, cooc_value, doc_quan, prev_doc_id;
} quad_int;

inline void FetchVocab(const char *path_to_vocab, std::unordered_map<std::string, int> &dictionary) {
  // If number of words in vocab is grater than 2^32, which is unlikely,
  // long long will be needed
  // This func reads word from vocab and sets a number for every
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

// ToDo: optimize copy in arr
inline void UploadBatchOnDisk(std::map<int, std::vector<quad_int>> &cooc) {
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
  const char *cname = name.c_str();
  // ToDo: create a folder for them
  FILE *out = fopen(cname, "wb");
  if (!out) {
    fprintf(stderr, "Failed to create a file in a working directory\n");
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
    if (needed_size > max_needed_size)
      max_needed_size = needed_size;
    arr[end_of_arr++] = needed_size / 3; // number of triples
    arr[end_of_arr++] = iter->first;
    for (auto iter2 = (iter->second).begin(); iter2 != (iter->second).end(); ++iter2) {
      arr[end_of_arr++] = iter2->token_id;
      arr[end_of_arr++] = iter2->cooc_value;
      arr[end_of_arr++] = iter2->doc_quan;
    }
  }
  fwrite(arr, sizeof *arr, end_of_arr, out);
  fclose(out);
  free(arr);
  batch_num++;
}

inline void form_quad(quad_int &tmp_quad, int second_token_id, int doc_id) {
  tmp_quad.token_id = second_token_id;
  tmp_quad.cooc_value = tmp_quad.doc_quan = 1;
  tmp_quad.prev_doc_id = doc_id;
}

inline void add_in_cooc_map(int first_token_id, int second_token_id,
        int doc_num, std::map<int, std::vector<quad_int>> &cooc_map) {
  quad_int tmp_quad;
  form_quad(tmp_quad, second_token_id, doc_num);
  std::vector<quad_int> tmp_vector;
  tmp_vector.push_back(tmp_quad);
  // As id of tokens which are inserted in map as a first parametr go in
  // ascending order way more often than in a way disturbing this order, it's
  // good to use hinted insertion.
  cooc_map.insert(cooc_map.end(), std::pair<int, std::vector<quad_int>>
          (first_token_id, tmp_vector));
}

inline void modify_cooc_map_node(int second_token_id, int doc_num,
        std::map<int, std::vector<quad_int>>::iterator map_record) {
  std::vector<quad_int> *vect_ptr = &map_record->second;
  int inserted_flag = 0;
  for (auto iter = vect_ptr->begin(); iter != vect_ptr->end(); ++iter) {
    if (iter->token_id == second_token_id) {
      iter->cooc_value++;
      if (iter->prev_doc_id != doc_num) {
        iter->prev_doc_id = doc_num;
        iter->doc_quan++;
      }
      inserted_flag = 1;
      break;
    }
  }
  if (!inserted_flag) {
    quad_int tmp_quad;
    form_quad(tmp_quad, second_token_id, doc_num);
    vect_ptr->push_back(tmp_quad);
  }
}

inline void ParseVowpalWabbitDoc(const char *path_to_wv, const
        std::unordered_map<std::string, int> &dictionary) {
  std::filebuf fb;
  if (!fb.open(path_to_wv, std::ios::in)) {
    fprintf(stderr, "Failed to open vowpal wabbit document\n");
    exit(1);
  }
  std::istream VowpalWabbitDoc(&fb);
  std::mutex lock;

  auto func = [&dictionary, &VowpalWabbitDoc, &lock]() {
    while (true) {
      std::vector<std::string> portion;

      {
        std::lock_guard<std::mutex> guard(lock);
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

      // The key in this map is first_token_id
      // The 1st elem of the tuple is token_id of token which occured
      // together with first token
      // The 2nd value is quantity of cooccurences in the collection
      // The 3rd is quantity of documents in which the following pair of
      // tokens occurred together
      // The 4th is number of the last doccument where the pair occurred

      std::map<int, std::vector<quad_int>> cooc_map;

      for (int doc_id = 0; doc_id < (int64_t) portion.size(); ++doc_id) {
        std::vector<std::string> doc;
        boost::split(doc, portion[doc_id], boost::is_any_of(" \t\r"));
        if (doc.size() <= 1)
          continue;
        for (int j = 1; j < (int64_t) doc.size(); ++j) {
          auto first_token = dictionary.find(doc[j]);
          if (first_token == dictionary.end())
            continue;
          int first_token_id = first_token->second;

          for (int k = 1; k < WINDOW_WIDTH && j + k < (int64_t) doc.size(); ++k) {
            auto second_token = dictionary.find(doc[j + k]);
            if (second_token == dictionary.end())
              continue;
            int second_token_id = second_token->second;
            int swap_flag = 0;
            if ( first_token_id > second_token_id) {
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
      if (!cooc_map.empty())
        UploadBatchOnDisk(cooc_map);
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
      exit(1);
    }
    ReadCellHeader(arr_batch, i);
  }
}

inline void FormListFromTriples(triple_int *addr, int triples_num, std::list<triple_int> &res) {
  for (int i = 0; i < triples_num; ++i)
    res.push_back(addr[i]);
}

inline void CheckCooccurrenceFreq(std::vector<std::tuple<int, int, std::list<triple_int>>> &result) {
  for (auto iter = std::get<LIST_OF_TRIPLES>(result[0]).begin();
           iter != std::get<LIST_OF_TRIPLES>(result[0]).end(); ) {
    if (iter->cooc_value < MIN_COOC_VALUE) {
      std::get<LIST_OF_TRIPLES>(result[0]).erase(iter++);
    } else
      iter++;
  }
  if (std::get<LIST_OF_TRIPLES>(result[0]).empty())
    result.pop_back();
}

// ToDo: Implement with sorted in ascending order forward list
inline void MergeListsWithAddition(triple_int *addr, int triples_num, std::list<triple_int> &tmp_list) {
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
        std::vector<std::tuple<int, int, std::list<triple_int>>> &res,
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

inline void ReadAndMergeBatches() {
  std::vector<batch_struct> arr_batch (batch_num);
  ArrBatchInitialization(arr_batch);
  std::make_heap(arr_batch.begin(), arr_batch.end(), BatchHeapComparator());
  FILE *res_file = CreateResFile();
  if (!res_file) {
    fprintf(stderr, "Failed to create a file in a working directory\n");
    exit(1);
  }

  // This vector won't hold more than 1 element, because another element
  // means another first_token_id => all the data linked with current
  // first_token_id can be filtered and be ready to be written in
  // resulting file.
  // The first element in the tuple is first_token_id.
  // The second is number of triples that are kept in the third element
  std::vector<std::tuple<int, int, std::list<triple_int>>> result;

  // From output_buf data will be loaded in resulting file
  // The first elem here is number of busy ints in buf.
  // The second is size of buf.
  // The third is pointer to buf
  std::pair<int, int *> output_buf;
  output_buf.second = (int *) malloc(OUTPUT_BUF_SIZE * sizeof *output_buf.second);

  // Data will be loaded in tmp_buf from batches and then into result vector
  int *tmp_buf = (int *) malloc(max_needed_size * sizeof(int));

  // Standard k-way merge as external sort
  while (!arr_batch.empty()) {
    // It's guaranteed that batches aren't empty (look ParseVowpalWabbitDoc func)
    std::list<triple_int> tmp_list;
    fread(tmp_buf, sizeof(triple_int), arr_batch[0].cell_size, arr_batch[0].file);
    if (result.empty()) {
      FormListFromTriples((triple_int *) tmp_buf, arr_batch[0].cell_size, tmp_list);
      result.push_back(std::tuple<int, int, std::list<triple_int>>
            (arr_batch[0].first_token_id, arr_batch[0].cell_size, tmp_list));
    } else if (std::get<FIRST_TOKEN_ID>(result[0]) == arr_batch[0].first_token_id) {
      MergeListsWithAddition((triple_int *) tmp_buf, arr_batch[0].cell_size, std::get<LIST_OF_TRIPLES>(result[0]));
      std::get<NUM_OF_TRIPLES>(result[0]) += arr_batch[0].cell_size;
    } else {
      CheckCooccurrenceFreq(result);
      if (!result.empty()) {
        DumpResRecordInBuf(output_buf, result, res_file);
        result.pop_back();
      }
      FormListFromTriples((triple_int *) tmp_buf, arr_batch[0].cell_size, tmp_list);
      result.push_back(std::tuple<int, int, std::list<triple_int>>
            (arr_batch[0].first_token_id, arr_batch[0].cell_size, tmp_list));
    }

    std::pop_heap(arr_batch.begin(), arr_batch.end(), BatchHeapComparator());
    if (ReadCellHeader(arr_batch, arr_batch.size() - 1))
      std::push_heap(arr_batch.begin(), arr_batch.end(), BatchHeapComparator());
    else {
      fclose(arr_batch[arr_batch.size() - 1].file);
      // ToDo: delete batch from disk here
      arr_batch.pop_back();
    }
  }
  CheckCooccurrenceFreq(result);
  if (!result.empty())
    DumpResRecordInBuf(output_buf, result, res_file);
  if (output_buf.first)
    WriteInResFile(output_buf, res_file);
  free(tmp_buf);
  free(output_buf.second);
  fclose(res_file);
}

// command line interface:
// ./main path/to/VowpalWabbitDoc path/to/vocab min_cooc_value
int main(int argc, char **argv) {
  // ToDo: replace native ptr
  // In unordered map we keep a pair: token and its id
  // If no co-occurrence found or it's too low an empty file is created
  std::unordered_map<string, int> dictionary;
  FetchVocab(argv[2], dictionary);
  ParseVowpalWabbitDoc(argv[1], dictionary);
  if (batch_num)
    ReadAndMergeBatches();
  else {
    FILE *res_file = CreateResFile();
    if (!res_file) {
      fprintf(stderr, "Failed to create a file in a working directory\n");
      exit(1);
    }
    fclose(res_file);
  }
  return 0;
}
