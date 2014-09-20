// Copyright 2011 Google Inc. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: nadavs@google.com <Nadav Samet>

#ifndef RPCZ_RPC_H
#define RPCZ_RPC_H

#include <stdexcept>
#include <string>

#include "rpcz/macros.hpp"
#include "rpcz/rpcz.pb.h"

namespace rpcz {

typedef rpc_response_header::status_code status_code;
typedef rpc_response_header::application_error_code application_error_code;

namespace status {
static const status_code INACTIVE = rpc_response_header::INACTIVE;
static const status_code ACTIVE = rpc_response_header::ACTIVE;
static const status_code OK = rpc_response_header::OK;
static const status_code CANCELLED = rpc_response_header::CANCELLED;
static const status_code APPLICATION_ERROR = rpc_response_header::APPLICATION_ERROR;
static const status_code DEADLINE_EXCEEDED = rpc_response_header::DEADLINE_EXCEEDED;
static const status_code TERMINATED = rpc_response_header::TERMINATED;
}  // namespace status
namespace application_error {
static const application_error_code RPCZ_NO_ERROR = rpc_response_header::RPCZ_NO_ERROR;
static const application_error_code INVALID_HEADER = rpc_response_header::INVALID_HEADER;
static const application_error_code NO_SUCH_SERVICE = rpc_response_header::NO_SUCH_SERVICE;
static const application_error_code NO_SUCH_METHOD = rpc_response_header::NO_SUCH_METHOD;
static const application_error_code INVALID_MESSAGE = rpc_response_header::INVALID_MESSAGE;
static const application_error_code METHOD_NOT_IMPLEMENTED = rpc_response_header::METHOD_NOT_IMPLEMENTED;
}  // namespace application_error

class sync_event;

class rpc {
 public:
  rpc();

  ~rpc();

  inline bool ok() const {
    return get_status() == status::OK;
  }

  status_code get_status() const {
    return status_;
  }

  inline std::string get_error_message() const {
    return error_message_;
  }

  inline int get_application_error_code() const {
    return application_error_code_;
  }

  inline int64 get_deadline_ms() const {
    return deadline_ms_;
  }

  inline void set_deadline_ms(int deadline_ms) {
    deadline_ms_ = deadline_ms;
  }

  void set_failed(int application_error_code, const std::string& message);

  int wait();

  std::string to_string() const;

 private:
  void set_status(status_code status);

  status_code status_;
  std::string error_message_;
  int application_error_code_;
  int64 deadline_ms_;
  scoped_ptr<sync_event> sync_event_;

  friend class rpc_channel_impl;
  friend class server_channel_impl;
  DISALLOW_COPY_AND_ASSIGN(rpc);
};

class rpc_error : public std::runtime_error {
 public:
  explicit rpc_error(const rpc& rpc_) 
      : std::runtime_error(rpc_.to_string()),
        status_(rpc_.get_status()),
        error_message_(rpc_.get_error_message()),
        application_error_code_(rpc_.get_application_error_code()) {}

  virtual ~rpc_error() throw() {}

  status_code get_status() const {
    return status_;
  }

  inline std::string get_error_message() const {
    return error_message_;
  }

  inline int get_application_error_code() const {
    return application_error_code_;
  }

 private:
  status_code status_;
  std::string error_message_;
  int application_error_code_;
};

class invalid_message_error : public std::runtime_error {
 public:
  explicit invalid_message_error(const std::string& message)
      : std::runtime_error(message) {}
};
}  // namespace
#endif
