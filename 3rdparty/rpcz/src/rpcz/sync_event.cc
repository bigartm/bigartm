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

#include "rpcz/sync_event.hpp"

namespace rpcz {

sync_event::sync_event() : ready_(false) {
}

void sync_event::wait() {
  boost::unique_lock<boost::mutex> lock(mu_);
  while (!ready_) {
    cond_.wait(lock);
  }
}

void sync_event::signal() {
  boost::unique_lock<boost::mutex> lock(mu_);
  ready_ = true;
  cond_.notify_all();
}
}  // namespace rpcz
