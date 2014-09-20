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

#include "boost/lexical_cast.hpp"
#include "rpcz/logging.hpp"
#include "rpcz/reactor.hpp"
#include "rpcz/rpc.hpp"
#include "rpcz/sync_event.hpp"
#include "rpcz/rpcz.pb.h"

namespace rpcz {

rpc::rpc()
    : status_(status::INACTIVE),
      application_error_code_(0),
      deadline_ms_(-1),
      sync_event_(new sync_event()) {
};

rpc::~rpc() {}

void rpc::set_failed(int application_error, const std::string& error_message) {
  set_status(status::APPLICATION_ERROR);
  error_message_ = error_message;
  application_error_code_ = application_error;
}

void rpc::set_status(status_code status) {
  status_ = status;
}

int rpc::wait() {
  status_code status = get_status();
  CHECK_NE(status, status::INACTIVE)
      << "Request must be sent before calling wait()";

  sync_event_->wait();
  return 0;
}

std::string rpc::to_string() const {
  std::string result =
      "status: " + rpc_response_header_status_code_Name(get_status());
  if (get_status() == status::APPLICATION_ERROR) {
    result += "(" + boost::lexical_cast<std::string>(
            get_application_error_code())
           + ")";
  }
  std::string error_message = get_error_message();
  if (!error_message.empty()) {
    result += ": " + error_message;
  }
  return result;
}
}  // namespace rpcz
