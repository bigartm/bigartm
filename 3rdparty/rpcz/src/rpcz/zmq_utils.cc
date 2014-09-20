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

#include "rpcz/zmq_utils.hpp"
#include <boost/functional/hash.hpp>
#include <sstream>
#include <stddef.h>
#include <string.h>
#include <zmq.h>
#include <ostream>
#include <string>
#include <vector>

#include "google/protobuf/stubs/common.h"
#include "zmq.hpp"
#include "rpcz/logging.hpp"
#include "rpcz/macros.hpp"

#ifdef _MSC_VER
#pragma warning( disable : 4018 )  // 'expression' : signed/unsigned mismatch
#endif

namespace rpcz {
std::string message_to_string(zmq::message_t& msg) {
  return std::string((char*)msg.data(), msg.size());
}

zmq::message_t* string_to_message(const std::string& str) {
  zmq::message_t* message = new zmq::message_t(str.length());
  memcpy(message->data(), str.c_str(), str.length());
  return message;
}

bool read_message_to_vector(zmq::socket_t* socket,
                            message_vector* data) {
  while (1) {
    zmq::message_t *msg = new zmq::message_t;
    socket->recv(msg, 0);
    int64_t more = 0;           //  Multipart detection
    size_t more_size = sizeof (more);
    socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
    data->push_back(msg);
    if (!more) {
      break;
    }
  }
  return true;
}

bool read_message_to_vector(zmq::socket_t* socket,
                         message_vector* routes,
                         message_vector* data) {
  bool first_part = true;
  while (1) {
    zmq::message_t *msg = new zmq::message_t;
    socket->recv(msg, 0);
    int64_t more = 0;           //  Multipart detection
    size_t more_size = sizeof(more);
    socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
    if (first_part) {
      routes->push_back(msg);
      if (msg->size() == 0) {
        first_part = false;
      }
    } else {
      data->push_back(msg);
    }
    if (!more) {
      return !first_part;
    }
  }
}

void write_vector_to_socket(zmq::socket_t* socket,
                            message_vector& data,
                            int flags) {
  for (size_t i = 0; i < data.size(); ++i) {
    socket->send(data[i], 
                 flags |
                 ((i < data.size() - 1) ? ZMQ_SNDMORE : 0));
  }
}

void write_vectors_to_socket(zmq::socket_t* socket,
                          message_vector& routes,
                          message_vector& data) {
  CHECK_GE(data.size(), 1u);
  write_vector_to_socket(socket, routes, ZMQ_SNDMORE);
  write_vector_to_socket(socket, data, 0);
}

bool send_empty_message(zmq::socket_t* socket,
                        int flags) {
  zmq::message_t message(0);
  return socket->send(message, flags);
}

bool send_string(zmq::socket_t* socket,
                 const std::string& str,
                 int flags) {
  zmq::message_t msg(str.size());
  str.copy((char*)msg.data(), str.size(), 0);
  return socket->send(msg, flags);
}

bool send_uint64(zmq::socket_t* socket,
                 google::protobuf::uint64 value,
                 int flags) {
  zmq::message_t msg(8);
  memcpy(msg.data(), &value, 8);
  return socket->send(msg, flags);
}

bool forward_message(zmq::socket_t &socket_in,
                     zmq::socket_t &socket_out) {
  message_vector routes;
  message_vector data;
  CHECK(!read_message_to_vector(&socket_in, &routes, &data));
  write_vector_to_socket(&socket_out, routes); 
  return true;
}

void log_message_vector(message_vector& vector) {
  LOG(INFO) << "---- " << static_cast<int>(vector.size()) << "----";
  boost::hash<std::string> hash;
  for (int i = 0; i < vector.size(); ++i) {
    std::string message(message_to_string(vector[i]));
    std::stringstream ss;
    ss << std::hex << hash(message);
    LOG(INFO) << "(" << static_cast<int>(vector[i].size()) << "): " << "[" << ss.str()
        << "]: "
        << message_to_string(vector[i]);
  }
  LOG(INFO) << "----------";
}
}  // namespace
