// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_COMMON_H_
#define SRC_ARTM_CORE_COMMON_H_

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <sstream>
#include <unordered_map>

#include "boost/uuid/uuid.hpp"

#include "glog/logging.h"

#include "artm/core/exceptions.h"
#include "artm/core/internals.pb.h"

namespace artm {
namespace core {

typedef std::string ModelName;
typedef std::string ScoreName;
typedef std::string RegularizerName;
typedef std::string TopicName;

typedef std::string ClassId;

struct Token {
 public:
  Token(ClassId _class_id, std::string _keyword)
      : keyword(_keyword), class_id(_class_id),
        hash_(std::hash<std::string>()(_keyword + _class_id)) {}

  Token& operator=(const Token &rhs) {
    if (this != &rhs) {
      const_cast<std::string&>(keyword) = rhs.keyword;
      const_cast<ClassId&>(class_id) = rhs.class_id;
      const_cast<size_t&>(hash_) = rhs.hash_;
    }

    return *this;
  }

  bool operator<(const Token& token) const {
    if (keyword != token.keyword)
      return keyword < token.keyword;
    return class_id < token.class_id;
  }

  bool operator==(const Token& token) const {
    if (keyword == token.keyword && class_id == token.class_id) return true;
    return false;
  }

  bool operator!=(const Token& token) const {
    return !(*this == token);
  }

  const std::string keyword;
  const ClassId class_id;

 private:
  friend struct TokenHasher;
  const size_t hash_;
};

struct TokenHasher {
  size_t operator()(const Token& token) const {
    return token.hash_;
  }
};

const std::string DefaultClass = "@default_class";

const int UnknownId = -1;

const std::string kBatchExtension = ".batch";

const int kIdleLoopFrequency = 1;  // 1 ms

class CuckooWatch {
 public:
  explicit CuckooWatch(std::string message)
      : message_(message), submessage_(), start_(std::chrono::system_clock::now()), parent_(nullptr),
        threshold_ms_(0) {}
  explicit CuckooWatch(std::string message, int threshold_ms)
      : message_(message), submessage_(), start_(std::chrono::system_clock::now()), parent_(nullptr),
        threshold_ms_(threshold_ms) {}
  CuckooWatch(std::string message, CuckooWatch* parent)
      : message_(message), submessage_(), start_(std::chrono::system_clock::now()), parent_(parent),
        threshold_ms_(1) {}
  CuckooWatch(std::string message, CuckooWatch* parent, int threshold_ms)
      : message_(message), submessage_(), start_(std::chrono::system_clock::now()), parent_(parent),
        threshold_ms_(threshold_ms) {}

  ~CuckooWatch() {
    auto delta = (std::chrono::system_clock::now() - start_);
    auto delta_ms = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
    if (delta_ms.count() < threshold_ms_)
      return;

    if (parent_ == nullptr) {
      std::stringstream ss;
      ss << delta_ms.count() << "ms in " << message_;
      if (!submessage_.empty())
        ss << " [including " << submessage_ << "]";
      LOG(INFO) << ss.str();
    } else {
      std::stringstream ss;
      ss << delta_ms.count() << "ms in " << message_ << "; ";
      parent_->submessage_ += ss.str();
    }
  }

 private:
  std::string message_;
  std::string submessage_;
  std::chrono::time_point<std::chrono::system_clock> start_;
  CuckooWatch* parent_;
  int threshold_ms_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_COMMON_H_
