// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <map>
#include <memory>

#include "boost/thread/locks.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"

#include "artm/core/common.h"

namespace artm {
namespace core {

// Singleton class to manage a collection of objects, identifiable with some integer ID.
template<class Type>
class TemplateManager : boost::noncopyable {
 public:
  static TemplateManager<Type>& singleton() {
    // Mayers singleton is thread safe in C++11
    // http://stackoverflow.com/questions/1661529/is-meyers-implementation-of-singleton-pattern-thread-safe
    static TemplateManager<Type> manager;
    return manager;
  }

  // Store an object and returns its ID.
  int Store(const Type& object) {
    boost::lock_guard<boost::mutex> guard(lock_);

    // iterate through instance_map_ until find an available slot
    while (map_.find(next_id_) != map_.end()) {
      next_id_++;
    }

    int id = next_id_++;

    map_.insert(std::make_pair(id, object));
    return id;
  }

  Type Get(int id) const {
    boost::lock_guard<boost::mutex> guard(lock_);
    auto iter = map_.find(id);
    return (iter == map_.end()) ? Type() : iter->second;
  }

  void Erase(int id) {
    // Destruction of the object should happen when no lock is acquired.
    // To achieve this we hold a reference to the object during 'map_.erase(id)'.
    // (think that Type is a shared_ptr<T>)
    Type object;

    {
      boost::lock_guard<boost::mutex> guard(lock_);
      auto iter = map_.find(id);
      if (iter != map_.end()) {
        object = iter->second;
      }

      map_.erase(id);
    }
  }

  void Clear() {
    boost::lock_guard<boost::mutex> guard(lock_);
    map_.clear();
  }

 private:
  // Singleton (make constructor private)
  TemplateManager() : lock_(), next_id_(1) { }

  mutable boost::mutex lock_;

  int next_id_;
  std::map<int, Type> map_;
};

}  // namespace core
}  // namespace artm
