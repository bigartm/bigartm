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

#include "rpcz/reactor.hpp"
#include <signal.h>
#include <vector>
#include "rpcz/callback.hpp"
#include "rpcz/clock.hpp"
#include "rpcz/logging.hpp"
#include "rpcz/macros.hpp"
#include "zmq.hpp"

#ifdef _MSC_VER
#pragma warning( disable : 4244 )  // 'argument' : conversion from 'T1' to 'T2'
#endif

namespace rpcz {
namespace {
static bool g_interrupted = false;
void signal_handler(int signal_value) {
  g_interrupted = true;
}
}  // unnamed namespace

reactor::reactor() : should_quit_(false) {
};

reactor::~reactor() {
  delete_container_pair_pointers(sockets_.begin(), sockets_.end());
  for (closure_run_map::const_iterator it = closure_run_map_.begin();
       it != closure_run_map_.end(); ++it) {
    delete_container_pointers(it->second.begin(), it->second.end());
  }
}

void reactor::add_socket(zmq::socket_t* socket, closure* closure) {
  sockets_.push_back(std::make_pair(socket, closure));
  is_dirty_ = true;
}

namespace {
void rebuild_poll_items(
    const std::vector<std::pair<zmq::socket_t*, closure*> >& sockets,
    std::vector<zmq::pollitem_t>* pollitems) {
  pollitems->resize(sockets.size());
  for (size_t i = 0; i < sockets.size(); ++i) {
    zmq::socket_t& socket = *sockets[i].first;
    zmq::pollitem_t pollitem = {socket, 0, ZMQ_POLLIN, 0};
    (*pollitems)[i] = pollitem;
  }
}
}  // namespace

void reactor::run_closure_at(uint64 timestamp, closure* closure) {
  closure_run_map_[timestamp].push_back(closure);
}

int reactor::loop() {
  while (!should_quit_) {
    if (is_dirty_) {
      rebuild_poll_items(sockets_, &pollitems_);
      is_dirty_ = false;
    }
    long poll_timeout = process_closure_run_map();
    int rc = zmq_poll(&pollitems_[0], pollitems_.size(), poll_timeout);

    if (rc == -1) {
      int zmq_err = zmq_errno();
      CHECK_NE(zmq_err, EFAULT);
      if (zmq_err == ETERM) {
        return -1;
      }
    }
    for (size_t i = 0; i < pollitems_.size(); ++i) {
      if (!pollitems_[i].revents & ZMQ_POLLIN) {
        continue;
      }
      pollitems_[i].revents = 0;
      sockets_[i].second->run();
    }
  }
  if (g_interrupted) {
    return -1;
  } else {
    return 0;
  }
}

long reactor::process_closure_run_map() {
  uint64 now = zclock_time();
  closure_run_map::iterator ub(closure_run_map_.upper_bound(now));
  for (closure_run_map::const_iterator it = closure_run_map_.begin();
       it != ub;
       ++it) {
    for (std::vector<closure*>::const_iterator vit = it->second.begin();
         vit != it->second.end(); ++vit) {
      (*vit)->run();
    }
  }
  long poll_timeout = -1;
  if (ub != closure_run_map_.end()) {
    poll_timeout = ub->first - now;
  }
  closure_run_map_.erase(closure_run_map_.begin(), ub);
  return poll_timeout;
}

void reactor::set_should_quit() {
  should_quit_ = true;
}

void install_signal_handler() {
#if !defined(_WIN32) && !defined(_WIN64)
  struct sigaction action;
  action.sa_handler = signal_handler;
  action.sa_flags = 0;
  sigemptyset(&action.sa_mask);
  sigaction(SIGINT, &action, NULL);
  sigaction(SIGTERM, &action, NULL);
#endif
}
}  // namespace rpcz
