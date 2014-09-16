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

#ifndef RPCZ_SERVER_H
#define RPCZ_SERVER_H

#include "rpcz/macros.hpp"
#include "rpcz/rpcz.pb.h"

namespace zmq {
class socket_t;
};

namespace rpcz {
class application;
class client_connection;
class connection_manager;
class message_iterator;
class rpc_service;
class server_channel;
class service;

// A server object maps incoming RPC requests to a provided service interface.
// The service interface methods are executed inside a worker thread.
class server {
 public:
  // Constructs a server that uses the provided application. The
  // application must outlive the server.
  explicit server(application& application);

  // Constructs a server that uses the provided connection_manager. The
  // connection_manager must outlive the server.
  explicit server(connection_manager& connection_manager);

  ~server();

  // Registers an rpc service with this server. All registrations must occur
  // before bind() is called. The name parameter identifies the service for
  // external clients. If you use the first form, the service name from the
  // protocol buffer definition will be used. Does not take ownership of the
  // provided service.
  void register_service(service* service);
  void register_service(service* service, const std::string& name);

  void bind(const std::string& endpoint);

  // Registers a low-level rpc_service.
  void register_service(rpc_service* rpc_service, const std::string& name);

 private:
  void handle_request(const client_connection& connection,
                      message_iterator& iter);

  connection_manager& connection_manager_;
  typedef std::map<std::string, rpcz::rpc_service*> rpc_service_map;
  rpc_service_map service_map_;
  DISALLOW_COPY_AND_ASSIGN(server);
};

// rpc_service is a low-level request handler: requests and replies are void*.
// It is exposed here for language bindings. Do not use directly.
class rpc_service {
 public:
  virtual ~rpc_service() {}

  virtual void dispatch_request(const std::string& method,
                               const void* payload, size_t payload_len,
                               server_channel* channel_) = 0;
};
}  // namespace
#endif
