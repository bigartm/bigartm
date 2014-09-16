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

#ifndef ZRPC_RPC_SERVICE_H
#define ZRPC_RPC_SERVICE_H

#include <string>
#include "Python.h"
#include "rpcz/rpcz.hpp"

namespace rpcz {

// A subclass of RpcService that helps forwarding the requests to Python-land.
class PythonRpcService : public rpc_service {
 public:
  typedef void(*Handler)(PyObject* user_data, std::string& method,
                         void* payload, size_t payload_len,
                         server_channel* channel);

  PythonRpcService(Handler handler, PyObject *user_data)
      : user_data_(user_data), handler_(handler) {
        Py_INCREF(user_data_);
  }

  ~PythonRpcService() {
    Py_DECREF(user_data_);
  }

  virtual void dispatch_request(const std::string& method,
                                const void* payload, size_t payload_len,
                                server_channel* channel) {
    handler_(user_data_,
             *const_cast<std::string*>(&method),
             const_cast<void*>(payload), payload_len, channel);
  }

 private:
  PyObject *user_data_;
  Handler handler_;
};
}  // namespace rpcz
#endif
