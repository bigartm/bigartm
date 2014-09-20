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

#ifndef RPCZ_APPLICATION_H
#define RPCZ_APPLICATION_H

#include <string>
#include "rpcz/macros.hpp"

namespace zmq {
class context_t; 
}  // namespace zmq

namespace rpcz {
class connection_manager;
class rpc_channel;
class server;

// rpcz::application is a simple interface that helps setting up a common
// RPCZ client or server application.
class application {
 public:
  class options {
   public:
    options() : connection_manager_threads(10),
                zeromq_context(NULL),
                zeromq_io_threads(1) {}
    options(int connection_manager_threads_size)
        : connection_manager_threads(connection_manager_threads_size),
          zeromq_context(NULL),
          zeromq_io_threads(1) {}

    // Number of connection manager threads. Those threads are used for
    // running user code: handling server requests or running callbacks.
    int connection_manager_threads;

    // ZeroMQ context to use for our application. If NULL, then application will
    // construct its own ZeroMQ context and own it. If you provide your own
    // ZeroMQ context, application will not take ownership of it. The ZeroMQ
    // context must outlive the application.
    zmq::context_t* zeromq_context;

    // Number of ZeroMQ I/O threads, to be passed to zmq_init(). This value is
    // ignored when you provide your own ZeroMQ context.
    int zeromq_io_threads;
  };

  application();

  explicit application(const options& options);

  virtual ~application();

  // Creates an rpc_channel to the given endpoint. Attach it to a Stub and you
  // can start making calls through this channel from any thread. No locking
  // needed. It is your responsibility to delete this object.
  virtual rpc_channel* create_rpc_channel(const std::string& endpoint);

  // Blocks the current thread until another thread calls terminate.
  virtual void run();

  // Releases all the threads that are blocked inside run()
  virtual void terminate();

 private:
  void init(const options& options);

  bool owns_context_;
  zmq::context_t* context_;
  scoped_ptr<connection_manager> connection_manager_;
  friend class server;
};
}  // namespace rpcz
#endif
