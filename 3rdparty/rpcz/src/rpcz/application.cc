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

#include <string>
#include <zmq.hpp>
#include "rpcz/application.hpp"
#include "rpcz/connection_manager.hpp"
#include "rpcz/rpc_channel.hpp"
#include "rpcz/server.hpp"

#ifdef _MSC_VER
#pragma warning( disable : 4355 )  // 'this' : used in base member initializer list
#endif

namespace rpcz {

application::application() {
  init(options());
};

application::application(const application::options& options) {
  init(options);
};

application::~application() {
  connection_manager_.reset();
  if (owns_context_) {
    delete context_;
  }
}

void application::init(const application::options& options) {
  if (options.zeromq_context) {
    context_ = options.zeromq_context;
    owns_context_ = false;
  } else {
    context_ = new zmq::context_t(options.zeromq_io_threads);
    owns_context_ = true;
  }
  connection_manager_.reset(new connection_manager(
          context_,
          options.connection_manager_threads));
}

rpc_channel* application::create_rpc_channel(const std::string& endpoint) {
  return rpc_channel::create(
      connection_manager_->connect(endpoint));
}

void application::run() {
  connection_manager_->run();
}

void application::terminate() {
  connection_manager_->terminate();
}
}  // namespace rpcz
