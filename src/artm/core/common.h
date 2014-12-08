// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_COMMON_H_
#define SRC_ARTM_CORE_COMMON_H_

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "boost/uuid/uuid.hpp"

#include "glog/logging.h"

#include "rpcz/rpc.hpp"

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

struct BatchManagerTask {
  BatchManagerTask(boost::uuids::uuid _uuid, std::string _file_path)
      : uuid(_uuid), file_path(_file_path) {}

  bool operator==(const BatchManagerTask& rhs) const {
    return uuid == rhs.uuid && file_path == rhs.file_path;
  }

  BatchManagerTask& operator=(const BatchManagerTask &rhs) {
    if (this != &rhs) {
      uuid = rhs.uuid;
      file_path = rhs.file_path;
    }
    return *this;
  }

  boost::uuids::uuid uuid;
  std::string file_path;
};

const std::string DefaultClass = "@default_class";

const int UnknownId = -1;

const std::string kBatchExtension = ".batch";

const int kIdleLoopFrequency = 1;  // 1 ms
const int kNetworkPollingFrequency = 50;  // 50 ms

class Notifiable {
 public:
  virtual ~Notifiable() {}
  virtual void Callback(ModelIncrement* model_increment) = 0;
};

inline bool make_rpcz_call(std::function<void()> f, const std::string& log_message = "", bool no_throw = false) {
  try {
    f();
    return true;
  } catch(const rpcz::rpc_error& error) {
    std::stringstream ss;

    if (error.get_status() == rpcz::status::DEADLINE_EXCEEDED) {
      ss << "Network comminication timeout";
      if (!log_message.empty()) ss << " in " << log_message;
      LOG(ERROR) << ss.str();
      if (!no_throw) throw artm::core::NetworkException(ss.str());
      return false;
    }

    if (error.get_status() == rpcz::status::APPLICATION_ERROR) {
      ss << "Remote RPCZ service application error";
      if (!log_message.empty()) ss << " in " << log_message;
      ss << ", code = " << error.get_application_error_code();
      if (!error.get_error_message().empty()) ss << ", error_message = " << error.get_error_message();
      LOG(ERROR) << ss.str();
      if (!no_throw) throw artm::core::NetworkException(ss.str());
      return false;
    }

    ss << "Network error";
    if (!log_message.empty()) ss << " in " << log_message;
    ss << ", rpcz_error_status = " << error.get_status();
    LOG(ERROR) << ss.str();
    if (!no_throw) throw artm::core::NetworkException(ss.str());
    return false;
  }

  return false;
}

inline bool make_rpcz_call_no_throw(std::function<void()> f, const std::string& log_message = "") {
  return make_rpcz_call(f, log_message, /*no_throw = */ true);
}

class CuckooWatch {
 public:
  explicit CuckooWatch(std::string message)
      : message_(message), start_(std::chrono::system_clock::now()) {}
  ~CuckooWatch() {
    auto delta = (std::chrono::system_clock::now() - start_);
    auto delta_ms = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
    LOG(INFO) << message_ << " " << delta_ms.count() << " milliseconds.";
  }

 private:
  std::string message_;
  std::chrono::time_point<std::chrono::system_clock> start_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_COMMON_H_
