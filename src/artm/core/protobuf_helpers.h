// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_PROTOBUF_HELPERS_H_
#define SRC_ARTM_CORE_PROTOBUF_HELPERS_H_

#include <string>
#include <vector>

#include "artm/core/common.h"
#include "artm/messages.pb.h"

namespace artm {
namespace core {

template<class T, class V>
bool repeated_field_contains(const T& field, V value) {
  for (int i = 0; i < field.size(); ++i) {
    if (field.Get(i) == value) {
      return true;
    }
  }

  return false;
}

template<class T, class V>
int repeated_field_index_of(const T& field, V value) {
  for (int i = 0; i < field.size(); ++i) {
    if (field.Get(i) == value) {
      return i;
    }
  }

  return -1;
}

inline bool model_has_token(const ::artm::TopicModel& topic_model,
                            const artm::core::Token& token) {
  for (int i = 0; i < topic_model.token_size(); ++i) {
    if (topic_model.token(i) == token.keyword &&
      topic_model.class_id(i) == token.class_id) return true;
  }

  return false;
}

template<class T, class V>
void repeated_field_append(T* field, int index, V value) {
  V new_value = field->Get(index) + value;
  field->Set(index, new_value);
}

template<class T>
std::vector<bool> is_member(T a, T b) {
  std::vector<bool> retval;

  if (a.size() > 0) {
    for (int i = 0; i < b.size(); ++i)
      retval.push_back(false);

    for (int i = 0; i < a.size(); ++i)
      for (int j = 0; j < b.size(); ++j)
        if (b.Get(j) == a.Get(i)) {
          retval[j] = true;
          break;
        }
  } else {
    for (int i = 0; i < b.size(); ++i)
      retval.push_back(true);
  }

  return retval;
}

template<class T, class V>
bool is_member(V a, T b) {
  for (int i = 0; i < b.size(); ++i) {
    if (b.Get(i) == a) {
      return true;
    }
  }

  return false;
}

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_PROTOBUF_HELPERS_H_
