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

#ifndef RPCZ_RPC_CHANNEL_IMPL_H
#define RPCZ_RPC_CHANNEL_IMPL_H

#include "rpcz/connection_manager.hpp"
#include "rpcz/rpc_channel.hpp"

namespace rpcz {

class connection;
class closure;
class message_vector;
struct rpc_response_context;

class rpc_channel_impl: public rpc_channel {
 public:
  rpc_channel_impl(connection connection);

  virtual ~rpc_channel_impl();

  virtual void call_method(const std::string& service_name,
                          const google::protobuf::MethodDescriptor* method,
                          const google::protobuf::Message& request,
                          google::protobuf::Message* response, rpc* rpc,
                          closure* done);

  virtual void call_method0(
      const std::string& service_name,
      const std::string& method_name,
      const std::string& request,
      std::string* response, rpc* rpc, closure* done);

 private:
  virtual void handle_client_response(
      rpc_response_context response_context, connection_manager::status status,
      message_iterator& iter);

  void call_method_full(
    const std::string& service_name,
    const std::string& method_name,
    const ::google::protobuf::Message* request_msg,
    const std::string& request,
    ::google::protobuf::Message* response_msg,
    std::string* response_str,
    rpc* rpc,
    closure* done);

  connection connection_;
};
} // namespace rpcz
#endif /* RPCZ_SIMPLE_RPC_CHANNEL_IMPL_H_ */
