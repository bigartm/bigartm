// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <queue>
#include <map>
#include <memory>
#include <vector>
#include <utility>

#include "boost/thread/locks.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"

#include "artm/core/common.h"

namespace artm {
namespace core {

// A helper-class, which magically turns any class into thread-safe thing.
// The only requirement: the class must have deep copy constructor.
// The key idea is in const get() method, which returns const shared_ptr<T>.
// This object can be further used without any locks, assuming that all
// access is read-only. In the meantime the object in ThreadSafeHolder
// might be replaced with a new instance (via set() method).
template<typename T>
class ThreadSafeHolder : boost::noncopyable {
 public:
  ThreadSafeHolder()
      : lock_(), object_(std::make_shared<T>()) { }

  explicit ThreadSafeHolder(const std::shared_ptr<T>& object)
      : lock_(), object_(object) { }

  ~ThreadSafeHolder() { }

  std::shared_ptr<T> get() const {
    boost::lock_guard<boost::mutex> guard(lock_);
    return object_;
  }

  std::shared_ptr<T> get_copy() const {
    boost::lock_guard<boost::mutex> guard(lock_);
    if (object_ == nullptr) {
      return std::make_shared<T>();
    }
    return std::make_shared<T>(*object_);
  }

  void set(const std::shared_ptr<T>& object) {
    boost::lock_guard<boost::mutex> guard(lock_);
    object_ = object;
  }

 private:
  mutable boost::mutex lock_;
  std::shared_ptr<T> object_;
};

template<typename K, typename T>
class ThreadSafeCollectionHolder : boost::noncopyable {
 public:
  ThreadSafeCollectionHolder()
      : lock_(), object_(std::map<K, std::shared_ptr<T>>()) { }

  static ThreadSafeCollectionHolder<K, T>& singleton() {
    // Mayers singleton is thread safe in C++11
    // http://stackoverflow.com/questions/1661529/is-meyers-implementation-of-singleton-pattern-thread-safe
    static ThreadSafeCollectionHolder<K, T> holder;
    return holder;
  }

  ~ThreadSafeCollectionHolder() { }

  std::shared_ptr<T> get(const K& key) const {
    boost::lock_guard<boost::mutex> guard(lock_);
    return get_locked(key);
  }

  bool has_key(const K& key) const {
    boost::lock_guard<boost::mutex> guard(lock_);
    return object_.find(key) != object_.end();
  }

  void erase(const K& key) {
    boost::lock_guard<boost::mutex> guard(lock_);
    auto iter = object_.find(key);
    if (iter != object_.end()) {
      object_.erase(iter);
    }
  }

  void clear() {
    boost::lock_guard<boost::mutex> guard(lock_);
    object_.clear();
  }

  std::shared_ptr<T> get_copy(const K& key) const {
    boost::lock_guard<boost::mutex> guard(lock_);
    auto value = get_locked(key);
    return value != nullptr ? std::make_shared<T>(*value) : std::shared_ptr<T>();
  }

  void set(const K& key, const std::shared_ptr<T>& object) {
    boost::lock_guard<boost::mutex> guard(lock_);
    auto iter = object_.find(key);
    if (iter != object_.end()) {
      iter->second = object;
    } else {
      object_.insert(std::pair<K, std::shared_ptr<T> >(key, object));
    }
  }

  std::vector<K> keys() const {
    boost::lock_guard<boost::mutex> guard(lock_);
    std::vector<K> retval;
    for (auto iter = object_.begin(); iter != object_.end(); ++iter) {
      retval.push_back(iter->first);
    }

    return retval;
  }

  size_t size() const {
    boost::lock_guard<boost::mutex> guard(lock_);
    return object_.size();
  }

  bool empty() const {
    boost::lock_guard<boost::mutex> guard(lock_);
    return object_.empty();
  }

 private:
  mutable boost::mutex lock_;
  std::map<K, std::shared_ptr<T> > object_;

  // Use this instead of get() when the lock is already acquired.
  std::shared_ptr<T> get_locked(const K& key) const {
    auto iter = object_.find(key);
    return (iter != object_.end()) ? iter->second : std::shared_ptr<T>();
  }
};

template<typename T>
class ThreadSafeQueue : boost::noncopyable {
 public:
  ThreadSafeQueue() : lock_(), queue_(), reserved_(0) { }

  bool try_pop(T* elem) {
    boost::lock_guard<boost::mutex> guard(lock_);
    if (queue_.empty()) {
      return false;
    }

    T tmp_elem = queue_.front();
    queue_.pop();
    *elem = tmp_elem;
    return true;
  }

  void push(const T& elem) {
    boost::lock_guard<boost::mutex> guard(lock_);
    queue_.push(elem);
  }

  void reserve() {
    boost::lock_guard<boost::mutex> guard(lock_);
    reserved_++;
  }

  void release() {
    boost::lock_guard<boost::mutex> guard(lock_);
    if (reserved_ > 0) {
      reserved_--;
    }
  }

  size_t size() const {
    boost::lock_guard<boost::mutex> guard(lock_);
    return queue_.size() + reserved_;
  }

  bool empty() const {
    boost::lock_guard<boost::mutex> guard(lock_);
    return queue_.empty();
  }

 private:
  mutable boost::mutex lock_;
  std::queue<T> queue_;
  size_t reserved_;
};

}  // namespace core
}  // namespace artm
