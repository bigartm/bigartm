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

#ifndef RPCZ_CONNECTION_MANAGER_H
#define RPCZ_CONNECTION_MANAGER_H

#include <string>
#include <boost/function.hpp>
#include <boost/thread.hpp>
#include "rpcz/macros.hpp"
#include "rpcz/sync_event.hpp"

namespace zmq {
class context_t;
class message_t;
class socket_t;
}  // namespace zmq

namespace rpcz {
class client_connection;
class closure;
class connection;
class connection_thread_context;
class message_iterator;
class message_vector;

// A connection_manager is a multi-threaded asynchronous system for communication
// over ZeroMQ sockets. A connection_manager can:
//   1. connect to a remote server and allow all threads the program to share
//      the connection:
//
//          connection_manager cm(10);
//          connection c = cm.connect("tcp://localhost:5557");
// 
//      Now, it is possible to send requests to this backend fron any thread:
//
//          c.send_request(...);
//
//  2. bind a socket and register a handler function. The handler function
//     gets executed in one of the connection manager threads.
//
//          c.bind("tcp://*:5555", &handler_function);
//
//  3. Queue closures to be executed on one of the connection manager threads:
//
//          c.add(closure)
//
// connection_manager and connection are thread-safe.
class connection_manager {
 public:
  enum status {
    INACTIVE = 0,
    ACTIVE = 1,
    DONE = 2,
    DEADLINE_EXCEEDED = 3,
  };

  typedef boost::function<void(const client_connection&, message_iterator&)>
      server_function;
  typedef boost::function<void(status status, message_iterator&)>
      client_request_callback;

  // Constructs a connection_manager that has nthreads worker threads. The
  // connection_manager does not take ownership of the given ZeroMQ context.
  connection_manager(zmq::context_t* context, int nthreads);

  // Blocks the current thread until all connection managers have completed.
  virtual ~connection_manager();

  // connects all connection_manager threads to the given endpoint. On success
  // this method returns a connection object that can be used from any thread
  // to communicate with this endpoint.
  virtual connection connect(const std::string& endpoint);

  // binds a socket to the given endpoint and registers server_function as a
  // handler for requests to this socket. The function gets executed on one of
  // the worker threads. When the function returns, the endpoint is already
  // bound.
  virtual void bind(const std::string& endpoint, server_function function);

  // Executes the closure on one of the worker threads.
  virtual void add(closure* closure);

  // Blocks this thread until terminate() is called from another thread.
  virtual void run();

  // Releases all the threads that are blocked inside run()
  virtual void terminate();

 private:
  zmq::context_t* context_;

  inline zmq::socket_t& get_frontend_socket();

  boost::thread broker_thread_;
  boost::thread_group worker_threads_;
  boost::thread_specific_ptr<zmq::socket_t> socket_;
  std::string frontend_endpoint_;
  sync_event is_termating_;

  DISALLOW_COPY_AND_ASSIGN(connection_manager);
  friend class connection;
  friend class client_connection;
  friend class connection_managerThread;
};

// Installs a SIGINT and SIGTERM handlers that causes all RPCZ's event loops
// to cleanly quit.
void install_signal_handler();

// Represents a connection to a server. Thread-safe.
class connection {
 public:
  connection() : manager_((connection_manager*)0xbadecafe), connection_id_(0) {}

  // Asynchronously sends a request over the connection.
  // request: a vector of messages to be sent. Does not take ownership of the
  //          request. The vector has to live valid at least until the request
  //          completes. It can be safely de-allocated inside the provided
  //          closure or after remote_response->wait() returns.
  // deadline_ms - milliseconds before giving up on this request. -1 means
  //               forever.
  // callback - a closure that will be ran on one of the worker threads when a
  //           response arrives or it timeouts.
  void send_request(
      message_vector& request,
      int64 deadline_ms,
      connection_manager::client_request_callback callback);

 private:
  connection(connection_manager *manager, uint64 connection_id) :
      manager_(manager), connection_id_(connection_id) {}
  connection_manager* manager_;
  uint64 connection_id_;
  friend class connection_manager;
};

class client_connection {
 public:
  void reply(message_vector* v);

 private:
  client_connection(connection_manager* manager, uint64 socket_id,
                   std::string& sender, std::string& event_id)
      : manager_(manager), socket_id_(socket_id), sender_(sender),
      event_id_(event_id) {}

  connection_manager* manager_;
  uint64 socket_id_;
  const std::string sender_;
  const std::string event_id_;
  friend void worker_thread(connection_manager*, zmq::context_t*, std::string);
};
}  // namespace rpcz
#endif
