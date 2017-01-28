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
  WINDOW_WIDTH = 5
};

// use boost to split words
void FetchVocab(const char *path_to_vocab, std::unordered_map<std::string, int> &dictionary) {
  // if number of words in vocab is grater than 2^32, which is unlikely,
  // long long will be needed
  int last_token_id = 0;
  FILE *vocab = fopen(path_to_vocab, "r");
  if (!vocab) {
    exit(1);
  }
  char *ptr = (char *) calloc(BLOCK_SIZE, sizeof *ptr);
  for (int bgn_pos = 0; ;) {
    int read_bytes = fread(ptr + bgn_pos, sizeof *ptr, BLOCK_SIZE - bgn_pos, vocab);
    if (!read_bytes) {
      if (bgn_pos) {
        ptr[bgn_pos] = '\0';
        std::string str = string(ptr);
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
        std::string str = string(ptr + prev_pos);
        dictionary.insert(std::make_pair(str, last_token_id++));
        prev_pos = i + 1;
      }
    }
    bgn_pos = 0;
    for (; prev_pos < read_bytes; ++bgn_pos)
      ptr[bgn_pos] = ptr[prev_pos];
  }
  free(ptr);
  fclose(vocab);
}

// create a class to make dependencies clearer
void form_quad(quad_int &tmp_quad, int second_token_id, int doc_id) {
  tmp_quad.token_id = second_token_id;
  tmp_quad.cooc_value = tmp_quad.doc_quan = 1;
  tmp_quad.prev_doc_id = doc_id;
}

void ParseVowpalWabbitDoc(const char *path_to_wv,
        const std::unordered_map<std::string, int> &dictionary) {
  std::filebuf fb;
  if (!fb.open(path_to_wv, std::ios::in)) {
    exit(1);
  }
  std::istream VowpalWabbitDoc(&fb);
  // The key in this map is first_token_id
  // The 1st elem of the tuple is token_id of token which occured together
  // with first token
  // The 2nd value is quantity of cooccurences in the collection
  // The 3rd is quantity of documents in which the following pair of tokens
  // occurred together
  // The 4th is number of the last doccument where the pair occurred
  typedef struct quad_int {
    int token_id, cooc_value, doc_quan, prev_doc_id;
  } quad_int;
  std::map<int, std::vector<quad_int>> cooccurrences;
  std::vector<std::string> portion;
  while(true) {
    for (int i = 0; i < ITEMS_PER_BATCH; ++i) {
      getline(VowpalWabbitDoc, portion[i]);
      if (VowpalWabbitDoc.eof())
        break;
    }
    for (int doc_id = 0; i < (int64_t) portion.size(); ++doc_id) {
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
          auto map_record = cooccurrences.find(first_token_id);
          // think about insertaion with constant time
          if (map_record == cooccurrences.end()) {
            quad_int tmp_quad;
            form_quad(tmp_quad, second_token_id, doc_id);
            std::vector<quad_int> tmp_vector;
            tmp_vector.push_back(tmp_quad);
            cooccurrences.insert(std::pair<int, std::vector<quad_int>>
                    (first_token_id, tmp_vector));
          } else {
            std::vector<quad_int> *vect_ptr = &map_record->second;
            int inserted_flag = 0;
            for (auto iter = vect_ptr->begin(); iter != vect_ptr->end(); ++iter) {
              if (iter->token_id == second_token_id) {
                iter->cooc_value++;
                if (iter->prev_doc_id != doc_id) {
                  iter->prev_doc_id = doc_id;
                  iter->doc_quan++;
                }
                inserted_flag = 1;
                break;
              }
            }
            if (!inserted_flag) {
              quad_int tmp_quad;
              form_quad(tmp_quad, second_token_id, doc_id);
              vect_ptr->push_back(tmp_quad);
            }
          }
        }
      }
    }
  }
}

// command line interface:
// ./main path/to/VowpalWabbitDoc path/to/vocab
int main(int argc, char **argv) {
  // throw an exception if file openning failed
  // In unordered map we keep a pair: token and its id
  std::unordered_map<string, int> dictionary;
  //FetchVocab(argv[2], dictionary);
  ParseVowpalWabbitDoc(argv[1], dictionary);
}
