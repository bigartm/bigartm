// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_COMMON_H_
#define SRC_ARTM_CORE_COMMON_H_

#include <functional>
#include <memory>
#include <string>

#include "glog/logging.h"

#include "rpcz/rpc.hpp"

#include "artm/core/exceptions.h"
#include "artm/core/internals.pb.h"

namespace artm {
namespace core {

typedef std::string ModelName;
typedef std::string ScoreName;
typedef std::string RegularizerName;

typedef std::string ClassId;

struct Token {
  Token(ClassId _class_id, std::string _keyword) : class_id(_class_id), keyword(_keyword) {}
  Token() : class_id(), keyword() {}

  void clear() {
    class_id.clear();
    keyword.clear();
  }

  bool operator<(const Token& token) const {
    if (class_id != token.class_id) {
      return class_id < token.class_id;
    }
    return keyword < token.keyword;
  }

  ClassId class_id;
  std::string keyword;
};

const std::string DefaultClass = "@default_class";

const int UnknownId = -1;

const std::string kBatchExtension = ".batch";

const int kIdleLoopFrequency = 1;  // 1 ms
const int kNetworkPollingFrequency = 50;  // 50 ms

class Notifiable {
 public:
  virtual ~Notifiable() {}
  virtual void Callback(std::shared_ptr<const ModelIncrement> model_increment) = 0;
};

inline void make_rpcz_call(std::function<void()> f, const std::string& f_name) {
  try {
    f();
  } catch(const rpcz::rpc_error&) {
    LOG(ERROR) << "Problems with connection between Proxy and NodeController in " <<
      f_name << "()";
    throw artm::core::NetworkException("Network error in function " + f_name + "()");
  }
}

inline void make_rpcz_call_no_throw(std::function<void()> f, const std::string& f_name) {
  try {
    f();
  } catch(const rpcz::rpc_error&) {
    LOG(ERROR) << "Problems with connection between Proxy and NodeController in " <<
      f_name << "()";
  }
}

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_COMMON_H_
