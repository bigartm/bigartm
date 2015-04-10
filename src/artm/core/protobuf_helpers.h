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
std::vector<bool> is_member(const T& elements, const T& set) {
  std::vector<bool> retval;
  retval.assign(set.size(), false);

  if (elements.size() > 0) {
    for (int j = 0; j < set.size(); ++j)
      for (int i = 0; i < elements.size(); ++i)
        if (set.Get(j) == elements.Get(i)) {
          retval[j] = true;
          break;
        }
  }

  return retval;
}

template<class T, class V>
bool is_member(const V& value, const T& set) {
  for (int i = 0; i < set.size(); ++i) {
    if (set.Get(i) == value) {
      return true;
    }
  }

  return false;
}

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_PROTOBUF_HELPERS_H_
