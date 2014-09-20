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

#include <boost/lexical_cast.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <glog/logging.h>
#include <stdio.h>
#include <zmq.hpp>
#include "gtest/gtest.h"
#include "rpcz/callback.hpp"
#include "rpcz/connection_manager.hpp"
#include "rpcz/macros.hpp"
#include "rpcz/sync_event.hpp"
#include "rpcz/zmq_utils.hpp"


namespace rpcz {

class connection_manager_test : public ::testing::Test {
 public:
  connection_manager_test() : context(1) {}

 protected:
  zmq::context_t context;
};

TEST_F(connection_manager_test, TestStartsAndFinishes) {
  connection_manager cm(&context, 4);
}

void echo_server(zmq::socket_t *socket) {
  bool should_quit = false;
  int messages = 0;
  while (!should_quit) {
    message_vector v;
    GOOGLE_CHECK(read_message_to_vector(socket, &v));
    ++messages;
    ASSERT_EQ(4, v.size());
    if (message_to_string(v[2]) == "hello") {
      ASSERT_EQ("there", message_to_string(v[3]).substr(0, 5));
    } else if (message_to_string(v[2]) == "QUIT") {
      should_quit = true;
    } else {
      GOOGLE_CHECK(false) << "Unknown command: " << message_to_string(v[2]);
    }
    write_vector_to_socket(socket, v);
  }
  delete socket;
}

boost::thread start_server(zmq::context_t* context) {
  zmq::socket_t* server = new zmq::socket_t(*context, ZMQ_DEALER);
  server->bind("inproc://server.test");
  return boost::thread(boost::bind(echo_server, server));
}

message_vector* create_simple_request(int number=0) {
  message_vector* request = new message_vector;
  request->push_back(string_to_message("hello"));
  char str[256];
  sprintf(str, "there_%d", number);
  request->push_back(string_to_message(str));
  return request;
}

message_vector* create_quit_request() {
  message_vector* request = new message_vector;
  request->push_back(string_to_message("QUIT"));
  request->push_back(string_to_message(""));
  return request;
}

void expect_timeout(connection_manager::status status, message_iterator& iter,
                   sync_event* sync) {
  ASSERT_EQ(connection_manager::DEADLINE_EXCEEDED, status);
  ASSERT_FALSE(iter.has_more());
  sync->signal();
}

TEST_F(connection_manager_test, TestTimeoutAsync) {
  zmq::socket_t server(context, ZMQ_DEALER);
  server.bind("inproc://server.test");

  connection_manager cm(&context, 4);
  connection connection(cm.connect("inproc://server.test"));
  scoped_ptr<message_vector> request(create_simple_request());

  sync_event event;
  connection.send_request(*request, 0,
                         boost::bind(&expect_timeout, _1, _2, &event));
  event.wait();
}

class barrier_closure : public connection_manager::client_request_callback {
 public:
  barrier_closure() : count_(0) {}

  void run(connection_manager::status status, message_iterator& iter) {
    boost::unique_lock<boost::mutex> lock(mutex_);
    ++count_;
    cond_.notify_all();
  }

  virtual void wait(int n) {
    boost::unique_lock<boost::mutex> lock(mutex_);
    while (count_ < n) {
      cond_.wait(lock);
    }
  }

 private:
  boost::mutex mutex_;
  boost::condition_variable cond_;
  int count_;
};

void SendManyMessages(connection connection, int thread_id) {
  boost::ptr_vector<message_vector> requests;
  const int request_count = 100;
  barrier_closure barrier;
  for (int i = 0; i < request_count; ++i) {
    message_vector* request = create_simple_request(
        thread_id * request_count * 17 + i);
    requests.push_back(request);
    connection.send_request(*request, -1,
                            boost::bind(&barrier_closure::run, &barrier, _1, _2));
  }
  barrier.wait(request_count);
}

TEST_F(connection_manager_test, ManyClientsTest) {
  boost::thread thread(start_server(&context));
  connection_manager cm(&context, 4);

  connection connection(cm.connect("inproc://server.test"));
  boost::thread_group group;
  for (int i = 0; i < 10; ++i) {
    group.add_thread(
        new boost::thread(boost::bind(SendManyMessages, connection, i)));
  }
  group.join_all();
  scoped_ptr<message_vector> request(create_quit_request());
  sync_event event;
  connection.send_request(*request, -1,
                         boost::bind(&sync_event::signal, &event));
  event.wait();
  thread.join();
}

void handle_request(client_connection connection,
                   message_iterator& request) {
  int value = boost::lexical_cast<int>(message_to_string(request.next()));
  message_vector v;
  v.push_back(string_to_message(boost::lexical_cast<std::string>(value + 1)));
  connection.reply(&v);
}

void handle_server_response(sync_event* sync,
                            connection_manager::status status,
                            message_iterator& iter) {
  CHECK_EQ(connection_manager::DONE, status);
  CHECK_EQ("318", message_to_string(iter.next()));
  sync->signal();
}

TEST_F(connection_manager_test, TestBindServer) {
  connection_manager cm(&context, 4);
  cm.bind("inproc://server.point", &handle_request);
  connection c = cm.connect("inproc://server.point");
  message_vector v;
  v.push_back(string_to_message("317"));
  sync_event event;
  c.send_request(v, -1,
                boost::bind(&handle_server_response, &event, _1, _2));
  event.wait();
}

const static char* kEndpoint = "inproc://test";
const static char* kReply = "gotit";

void DoThis(zmq::context_t* context) {
  LOG(INFO)<<"Creating socket. Context="<<context;
  zmq::socket_t socket(*context, ZMQ_PUSH);
  socket.connect(kEndpoint);
  send_string(&socket, kReply);
  socket.close();
  LOG(INFO)<<"socket closed";
}

TEST_F(connection_manager_test, ProcessesSingleCallback) {
  connection_manager cm(&context, 4);
  zmq::socket_t socket(context, ZMQ_PULL);
  socket.bind(kEndpoint);
  cm.add(new_callback(&DoThis, &context));
  message_vector messages;
  CHECK(read_message_to_vector(&socket, &messages));
  ASSERT_EQ(1, messages.size());
  CHECK_EQ(kReply, message_to_string(messages[0]));
}

void Increment(boost::mutex* mu,
               boost::condition_variable* cond, int* x) {
  mu->lock();
  (*x)++;
  cond->notify_one();
  mu->unlock();
}

void add_many_closures(connection_manager* cm) {
  boost::mutex mu;
  boost::condition_variable cond;
  boost::unique_lock<boost::mutex> lock(mu);
  int x = 0;
  const int kMany = 137;
  for (int i = 0; i < kMany; ++i) {
    cm->add(new_callback(&Increment, &mu, &cond, &x));
  }
  CHECK_EQ(0, x);  // since we are holding the lock
  while (x != kMany) {
    cond.wait(lock);
  }
}

TEST_F(connection_manager_test, ProcessesManyCallbacksFromManyThreads) {
  const int thread_count = 10;
  connection_manager cm(&context, thread_count);
  boost::thread_group thread_group;
  for (int i = 0; i < thread_count; ++i) {
    thread_group.add_thread(
        new boost::thread(boost::bind(add_many_closures, &cm)));
  }
  thread_group.join_all();
}
}  // namespace rpcz
