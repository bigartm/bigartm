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

#include "rpcz/server.hpp"
#include <signal.h>
#include <string.h>
#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/errno.h>
#include <sys/signal.h>
#endif
#include <functional>
#include <utility>

#include <boost/bind.hpp>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>
#include <google/protobuf/stubs/common.h>
#include <zmq.hpp>

#include "rpcz/application.hpp"
#include "rpcz/callback.hpp"
#include "rpcz/connection_manager.hpp"
#include "rpcz/logging.hpp"
#include "rpcz/macros.hpp"
#include "rpcz/rpc.hpp"
#include "rpcz/reactor.hpp"
#include "rpcz/service.hpp"
#include "rpcz/zmq_utils.hpp"
#include "rpcz/rpcz.pb.h"

namespace rpcz {

class server_channel_impl : public server_channel {
 public:
  server_channel_impl(const client_connection& connection)
      : connection_(connection) {
      }

  virtual void send(const google::protobuf::Message& response) {
    rpc_response_header generic_rpc_response;
    int msg_size = response.ByteSize();
    scoped_ptr<zmq::message_t> payload(new zmq::message_t(msg_size));
    if (!response.SerializeToArray(payload->data(), msg_size)) {
      throw invalid_message_error("Invalid response message");
    }
    send_generic_response(generic_rpc_response,
                        payload.release());
  }

  virtual void send0(const std::string& response) {
    rpc_response_header generic_rpc_response;
    send_generic_response(generic_rpc_response,
                        string_to_message(response));
  }

  virtual void send_error(int application_error,
                          const std::string& error_message="") {
    rpc_response_header generic_rpc_response;
    zmq::message_t* payload = new zmq::message_t();
    generic_rpc_response.set_status(status::APPLICATION_ERROR);
    generic_rpc_response.set_application_error(application_error);
    if (!error_message.empty()) {
      generic_rpc_response.set_error(error_message);
    }
    send_generic_response(generic_rpc_response,
                        payload);
  }

 private:
  client_connection connection_;
  scoped_ptr<google::protobuf::Message> request_;

  // Sends the response back to a function server through the reply function.
  // Takes ownership of the provided payload message.
  void send_generic_response(const rpc_response_header& generic_rpc_response,
                           zmq::message_t* payload) {
    size_t msg_size = generic_rpc_response.ByteSize();
    zmq::message_t* zmq_response_message = new zmq::message_t(msg_size);
    CHECK(generic_rpc_response.SerializeToArray(
            zmq_response_message->data(),
            msg_size));

    message_vector v;
    v.push_back(zmq_response_message);
    v.push_back(payload);
    connection_.reply(&v);
  }

  friend class proto_rpc_service;
};

class proto_rpc_service : public rpc_service {
 public:
  explicit proto_rpc_service(service* service) : service_(service) {
  }

  virtual void dispatch_request(const std::string& method,
                               const void* payload, size_t payload_len,
                               server_channel* channel_) {
    scoped_ptr<server_channel_impl> channel(
        static_cast<server_channel_impl*>(channel_));

    const ::google::protobuf::MethodDescriptor* descriptor =
        service_->GetDescriptor()->FindMethodByName(
            method);
    if (descriptor == NULL) {
      // Invalid method name
      DLOG(INFO) << "Invalid method name: " << method,
      channel->send_error(application_error::NO_SUCH_METHOD);
      return;
    }
    channel->request_.reset(CHECK_NOTNULL(
            service_->GetRequestPrototype(descriptor).New()));
    if (!channel->request_->ParseFromArray(payload, payload_len)) {
      DLOG(INFO) << "Failed to parse request.";
      // Invalid proto;
      channel->send_error(application_error::INVALID_MESSAGE);
      return;
    }
    server_channel_impl* channel_ptr = channel.release();
    service_->call_method(descriptor,
                         *channel_ptr->request_,
                         channel_ptr);
  }

 private:
  scoped_ptr<service> service_;
};

server::server(application& application)
  : connection_manager_(*application.connection_manager_.get()) {
}

server::server(connection_manager& connection_manager)
  : connection_manager_(connection_manager) {
}

server::~server() { }

void server::register_service(rpcz::service *service) {
  register_service(service,
                  service->GetDescriptor()->name());
}

void server::register_service(rpcz::service *service, const std::string& name) {
  register_service(new proto_rpc_service(service),
                  name);
}

void server::register_service(rpcz::rpc_service *rpc_service,
                              const std::string& name) {
  service_map_[name] = rpc_service;
}

void server::bind(const std::string& endpoint) {
  connection_manager::server_function f = boost::bind(
      &server::handle_request, this, _1, _2);
  connection_manager_.bind(endpoint, f);
}

void server::handle_request(const client_connection& connection,
                            message_iterator& iter) {
  if (!iter.has_more()) {
    return;
  }
  rpc_request_header rpc_request_header;
  scoped_ptr<server_channel> channel(new server_channel_impl(connection));
  {
    zmq::message_t& msg = iter.next();
    if (!rpc_request_header.ParseFromArray(msg.data(), msg.size())) {
      // Handle bad rpc.
      DLOG(INFO) << "Received bad header.";
      channel->send_error(application_error::INVALID_HEADER);
      return;
    };
  }
  if (!iter.has_more()) {
    return;
  }
  zmq::message_t& payload = iter.next();
  if (iter.has_more()) {
    return;
  }

  rpc_service_map::const_iterator service_it = service_map_.find(
      rpc_request_header.service());
  if (service_it == service_map_.end()) {
    // Handle invalid service.
    DLOG(INFO) << "Invalid service: " << rpc_request_header.service();
    channel->send_error(application_error::NO_SUCH_SERVICE);
    return;
  }
  rpcz::rpc_service* service = service_it->second;
  service->dispatch_request(rpc_request_header.method(),
                           payload.data(), payload.size(),
                           channel.release());
}
}  // namespace
