// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <stdint.h>

#include <vector>
#include <unordered_map>

namespace artm {
namespace utility {

template<class T>
inline int64_t getMemoryUsage(const std::vector<T>& obj) {
  return sizeof(obj) + sizeof(T) * obj.capacity();
}

template<>
inline int64_t getMemoryUsage(const std::vector<bool>& obj) {
  return sizeof(obj) + obj.capacity() / 8;
}

template<class K, class V, class H>
inline int64_t getMemoryUsage(const std::unordered_map<K, V, H>& obj) {
  int64_t entrySize = sizeof(K) + sizeof(V) + sizeof(void*);
  int64_t bucketSize = sizeof(void*);
  int64_t adminSize = 3 * sizeof(void*) + sizeof(size_t);
  return adminSize + obj.size() * entrySize + obj.bucket_count() * bucketSize;
}

}  // namespace utility
}  // namespace artm
