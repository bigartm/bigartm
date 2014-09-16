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

#ifndef RPCZ_REACTOR_H
#define RPCZ_REACTOR_H

#include <map>
#include <vector>
#include "zmq.hpp"
#include "rpcz/macros.hpp"

namespace rpcz {

class closure;

class reactor {
 public:
  reactor();
  ~reactor();

  void add_socket(zmq::socket_t* socket, closure* callback);

  void run_closure_at(uint64 timestamp, closure *callback);

  int loop();

  void set_should_quit();

 private:
  long process_closure_run_map();

  bool should_quit_;
  bool is_dirty_;
  std::vector<std::pair<zmq::socket_t*, closure*> > sockets_;
  std::vector<zmq::pollitem_t> pollitems_;
  typedef std::map<uint64, std::vector<closure*> > closure_run_map;
  closure_run_map closure_run_map_;
  DISALLOW_COPY_AND_ASSIGN(reactor);
};
}  // namespace
#endif
