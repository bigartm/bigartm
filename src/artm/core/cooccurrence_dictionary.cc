#include <cstdio>
#include <unordered_map>
#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <map>
#include <vector>
#include <list>
#include "boost/algorithm/string.hpp"

using namespace std;

// ToDo: replace it with variables
enum {
  BLOCK_SIZE = 65536,
  ITEMS_PER_BATCH = 2 /*10000*/,
  WINDOW_WIDTH = 5,
  BATCH_STARTING_SIZE = 65536,
  CELL_STARTING_SIZE = 4096,
  MAX_NAME_LEN = 30,
  OUTPUT_BUF_SIZE = 67108864, // 2 ^ 26
  MIN_COOC_VALUE = 0,
};

static int batch_num = 0;
static int max_needed_size = 0;

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
  // 2. Vocab file mustn't contain lines, that consist of tabs or spaces
  // only (otherwise undefined behavior)

  std::string str;
  int last_token_id = 0;
  FILE *vocab = fopen(path_to_vocab, "r");
  if (!vocab)
    exit(1);
  char *ptr = (char *) calloc(BLOCK_SIZE, sizeof *ptr);

  for (int bgn_pos = 0; ;) {
    int read_items = (int) fread(ptr + bgn_pos, sizeof *ptr, BLOCK_SIZE - bgn_pos, vocab);
    if (!read_items) {
      if (bgn_pos) {
        ptr[bgn_pos] = '\0';
        str = string(ptr);
        dictionary.insert(std::make_pair(str, last_token_id++));
      }
      break;
    }
    int prev_pos = 0;
    for (int i = 0; i < bgn_pos + read_items; ++i) {
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
    for (; prev_pos < read_items + prev_bgn_pos; ++bgn_pos, ++prev_pos)
      ptr[bgn_pos] = ptr[prev_pos];
  }
  free(ptr);
  fclose(vocab);
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
// ToDo: make posibility to work with vector of triples (polymorphism), and to
// adjust the name if it's specified as an argument
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
  if (!out)
    exit(1);
  int arr_size = BATCH_STARTING_SIZE;
  int *arr = (int *) malloc(arr_size * sizeof *arr);
  int end_of_arr = 0;

  for (auto iter = cooc.begin(); iter != cooc.end(); ++iter) {
    int needed_size = (iter->second).size() * 3;
    while (needed_size + 2 > arr_size - end_of_arr) {
      arr_size <<= 1;
      arr = (int *) realloc(arr, arr_size * sizeof *arr);
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
    if (!arr_batch[i].file)
      exit(1);
    ReadCellHeader(arr_batch, i);
  }
}

inline void FormListFromTriples(triple_int *addr, int triples_num, std::list<triple_int> &res) {
  for (int i = 0; i < triples_num; ++i)
    res.push_back(addr[i]);
}

inline void CheckCooccurrenceFreq(std::list<std::pair<int, std::list<triple_int>>> &result) {
  auto tmp_iter = result.rend();
  tmp_iter++;
  for (auto iter = (tmp_iter->second).begin(); iter != (tmp_iter->second).end(); ) {
    if (iter->cooc_value < MIN_COOC_VALUE) {
      (tmp_iter->second).erase(iter++);
    } else
      iter++;
  }
  if ((tmp_iter->second).empty())
    result.pop_back();
}

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

inline FILE *CreateResFile() {
  return fopen("Co-occurrenceDictionary.txt", "r");
}

// ToDo: optimize - write forward list by yourself
inline void ReadAndMergeBatches() {
  std::vector<batch_struct> arr_batch (batch_num);
  ArrBatchInitialization(arr_batch);
  std::make_heap(arr_batch.begin(), arr_batch.end(), BatchHeapComparator());

  // Standard k-way merge as external sort

  int end_of_buf = 0;

  // From output_buf data will be loaded in resulting file
  //int *output_buf = (int *) malloc(OUTPUT_BUF_SIZE * sizeof *output_buf);

  // Data will be load in tmp_buf from batches and then into a forward list
  int *tmp_buf = (int *) malloc(max_needed_size * sizeof(int));

  std::list<std::pair<int, std::list<triple_int>>> result;
  while (!arr_batch.empty()) {
    // It's guaranteed that batches aren't empty (look ParseVowpalWabbitDoc func)
    fread(tmp_buf, sizeof(triple_int), arr_batch[0].cell_size, arr_batch[0].file);

    auto rev_iter = result.rend();
    rev_iter++;
    std::list<triple_int> tmp_list;
    if (result.empty()) {
      FormListFromTriples((triple_int *) tmp_buf, arr_batch[0].cell_size, tmp_list);
      result.push_back(std::pair<int, std::list<triple_int>>
              (arr_batch[0].first_token_id, tmp_list));
    } else if (rev_iter->first == arr_batch[0].first_token_id) {
      MergeListsWithAddition((triple_int *) tmp_buf, arr_batch[0].cell_size, rev_iter->second);
    } else {
      CheckCooccurrenceFreq(result);
      FormListFromTriples((triple_int *) tmp_buf, arr_batch[0].cell_size, tmp_list);
      result.push_back(std::pair<int, std::list<triple_int>>
              (arr_batch[0].first_token_id, tmp_list));
    }

    std::pop_heap(arr_batch.begin(), arr_batch.end(), BatchHeapComparator());
    if (ReadCellHeader(arr_batch, arr_batch.size() - 1))
      std::push_heap(arr_batch.begin(), arr_batch.end(), BatchHeapComparator());
    else {
      fclose(arr_batch[arr_batch.size() - 1].file);
      arr_batch.pop_back();
    }
    // ToDo: delete batch from disk here
    // ToDo: define when to dump whole list in buf
  }
  //CreateResFile();
  //UploadResultOnDisk();
  //free(tmp_buf);
  //free(output_buf);
}

// command line interface:
// ./main path/to/VowpalWabbitDoc path/to/vocab
int main(int argc, char **argv) {
  // ToDo: throw exceptions in places where exit(1) is or write to stderr
  // ToDo: add threads
  // ToDo: replace native ptr
  // In unordered map we keep a pair: token and its id
  // If no co-occurrence found or it's too low an empty file is created
  std::unordered_map<string, int> dictionary;
  FetchVocab("vocab", dictionary);
  ParseVowpalWabbitDoc("VW", dictionary);
  if (batch_num)
    ReadAndMergeBatches();
  /*else {
    FILE *res_file = CreateResFile();
    fclose(res_file);
  }*/
  return 0;
}
