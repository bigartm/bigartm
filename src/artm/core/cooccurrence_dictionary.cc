#include <cstdio>
#include <unordered_map>
#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <map>
#include <vector>

#include "boost/algorithm/string.hpp"

using namespace std;

enum {
  BLOCK_SIZE = 65536,
  ITEMS_PER_BATCH = 10000,
  WINDOW_WIDTH = 5,
  BATCH_STARTING_SIZE = 65536,
  CELL_STARTING_SIZE = 4096,
  MAX_LEN_NAME = 20,
};

static int batch_num = 0;

typedef struct triple_int {
  int token_id, cooc_value, doc_quan;
} triple_int;

typedef struct quad_int {
  int token_id, cooc_value, doc_quan, prev_doc_id;
} quad_int;

// ToDo: fix ub
inline void FetchVocab(const char *path_to_vocab, std::unordered_map<std::string, int> &dictionary) {

  // 1. If number of words in vocab is grater than 2^32, which is unlikely,
  // long long will be needed
  // 2. Vocab file mustn't contain lines, that consist from tabs or spaces
  // only (otherwise undefined behavior)

  std::string str;
  int last_token_id = 0;
  FILE *vocab = fopen(path_to_vocab, "r");
  if (!vocab)
    exit(1);
  char *ptr = (char *) calloc(BLOCK_SIZE, sizeof *ptr);

  for (int bgn_pos = 0; ;) {
    int read_bytes = (int) fread(ptr + bgn_pos, sizeof *ptr, BLOCK_SIZE - bgn_pos, vocab);
    if (!read_bytes) {
      if (bgn_pos) {
        ptr[bgn_pos] = '\0';
        str = string(ptr);
        dictionary.insert(std::make_pair(str, last_token_id++));
      }
      break;
    }
    int prev_pos = 0;
    for (int i = 0; i < bgn_pos + read_bytes; ++i) {
      if (ptr[i] == '\n') {
        if (i == prev_pos) {
          prev_pos = i + 1;
            continue;
        }
        if (ptr[i - 1] == '\r') {
          if (i - 1 == prev_pos) {
            prev_pos = i + 1;
            continue;
          }
          ptr[i - 1] = '\0';
        } else
          ptr[i] = '\0';
        str = string(ptr + prev_pos);
        dictionary.insert(std::make_pair(str, last_token_id++));
        prev_pos = i + 1;
      }
    }
    int prev_bgn_pos = bgn_pos;
    bgn_pos = 0;
    for (; prev_pos < read_bytes + prev_bgn_pos; ++bgn_pos, ++prev_pos)
      ptr[bgn_pos] = ptr[prev_pos];
  }
  free(ptr);
  fclose(vocab);
}

inline void FormName(int num, std::string &name) {
  char num_str[20] = "";
  sprintf(num_str,"%d", num);
  name = string("cooccurrence");
  std::string str2 = string(num_str);
  std::string str3 = string(".bin");
  name += str2 + str3;
}

// ToDo: optimize copy in arr
// ToDo: make posibility to work with vector of triples, and to
// adjust the name if it's specified as an argument
inline void UploadBatchOnDisk(std::map<int, std::vector<quad_int>> &cooc) {
  // This function creates in a special directory a binary file wth all
  // content from the map (batch)
  std::string name;
  FormName(batch_num, name);
  const char *cname = name.c_str();
  // ToDo: create a folder for them
  FILE *out = fopen(cname, "wb");
  if (!out)
    exit(1);
  int arr_size = BATCH_STARTING_SIZE;
  int *arr = (int *) malloc(arr_size * sizeof *arr);
  int end_of_arr = 0;

  // Every node of the map is stored in a cell as seqence of ints (4 bytes)
  // First elem of a cell is size of rest of a cell
  // Second element is token_id of the first token
  // Then go triples, every triple consist of token id of the second token,
  // cooc_value and quantity of documents in which the folowing pair of
  // tokens occurred together

  for (auto iter = cooc.begin(); iter != cooc.end(); ++iter) {
    int needed_size = (iter->second).size() * 3 + 2;
    if (needed_size > arr_size - end_of_arr + 1) {
      arr_size <<= 1;
      arr = (int *) realloc(arr, arr_size * sizeof *arr);
    }
    arr[end_of_arr++] = (needed_size - 1) * sizeof(int);
    arr[end_of_arr++] = iter->first;
    for (auto iter2 = (iter->second).begin();
             iter2 != (iter->second).end(); ++iter2) {
      arr[end_of_arr++] = iter2->token_id;
      arr[end_of_arr++] = iter2->cooc_value;
      arr[end_of_arr++] = iter2->doc_quan;
    }
  }
  arr[end_of_arr] = 0;
  fwrite(arr, sizeof *arr, end_of_arr, out);
  fclose(out);
  free(arr);
  batch_num++;
}

inline void UploadResultOnDisk(const std::map<int, std::vector<triple_int>> &merged_batch) {
  ;
}

// ToDo: create a class to make dependencies clearer
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
  cooc_map.insert(std::pair<int, std::vector<quad_int>> (first_token_id, tmp_vector));
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
  if (!fb.open(path_to_wv, std::ios::in))
    exit(1);
  std::istream VowpalWabbitDoc(&fb);

  // ToDo: invent a new way to stop the loop
  int portion_size = ITEMS_PER_BATCH;
  for (int portion_num = 0, global_doc_num = 0; portion_size ==
          ITEMS_PER_BATCH; ++portion_num, global_doc_num += ITEMS_PER_BATCH) {
    std::vector<std::string> portion(ITEMS_PER_BATCH);
    for (portion_size = 0; portion_size < ITEMS_PER_BATCH; ++portion_size) {
      getline(VowpalWabbitDoc, portion[portion_size]);
      if (VowpalWabbitDoc.eof())
        break;
    }

    // The key in this map is first_token_id
    // The 1st elem of the tuple is token_id of token which occured together
    // with first token
    // The 2nd value is quantity of cooccurences in the collection
    // The 3rd is quantity of documents in which the following pair of tokens
    // occurred together
    // The 4th is number of the last doccument where the pair occurred

    std::map<int, std::vector<quad_int>> cooc_map;

    for (int doc_id = 0; doc_id < (int64_t) portion_size; ++doc_id) {
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
          // ToDo: think about insertion with constant time
          if (map_record == cooc_map.end())
            add_in_cooc_map(first_token_id, second_token_id, global_doc_num + doc_id, cooc_map);
          else
            modify_cooc_map_node(second_token_id, global_doc_num + doc_id, map_record);
          if (swap_flag) {
            first_token_id = second_token_id;
          }
        }
      }
    }
    if (!cooc_map.empty())
      UploadBatchOnDisk(cooc_map);
  }
}

// ToDo: delete batches
inline void ReadAndMergeBatches(const std::unordered_map<string, int> &dictionary) {
  typedef struct batch_struct {
    int storage_size; // how many elements, not bytes
    int cell_size; // size of useful memory
    int *storage;
    FILE *file;
  } batch_struct;

  std::vector<batch_struct> arr_batches (batch_num);

  // Here storage is a pointer to memory which will keep one cell from a
  // batch. Memory all the cells will be allocked once and later in a
  // case of lack it'll be increased (realloc).
  // Plus of this allocation is that calls of malloc and realloc will happen
  // very few times.
  // Minus: internal fragmentation could be high enough.
  // Binary files (batches) must be safe, because of the size value kept in
  // there. That is how many bytes are needed to be read. Not more, not less.

  for (int i = 0; i < batch_num; ++i) { // initialization
    std::string name;
    FormName(i, name);
    arr_batches[i].file = fopen(name.c_str(), "rb");
    if (!arr_batches[i].file)
      exit(1);
    int size = CELL_STARTING_SIZE;
    int needed_size = 0;
    fread(&needed_size, sizeof(int), 1, arr_batches[i].file);
    arr_batches[i].cell_size = needed_size;
    while (size < needed_size)
      size <<= 1;
    arr_batches[i].storage_size = size;
    arr_batches[i].storage = (int *) malloc(size * sizeof(int));
    fread(arr_batches[i].storage + 1, sizeof(int), needed_size / sizeof(int), arr_batches[i].file);
  }
  std::map<int, std::vector<triple_int>> merged_batch;

  ;

  UploadResultOnDisk(merged_batch);
}

// command line interface:
// ./main path/to/VowpalWabbitDoc path/to/vocab
int main(int argc, char **argv) {
  // ToDo: throw exceptions in places where are exit(1) or write to stderr
  // ToDo: add threads
  // In unordered map we keep a pair: token and its id
  std::unordered_map<string, int> dictionary;
  FetchVocab("vocab", dictionary);
  ParseVowpalWabbitDoc("VW", dictionary);
  ReadAndMergeBatches(dictionary);
  return 0;
}
