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

#include "artm/core/sync_event.h"

namespace artm {
namespace core {

SyncEvent::SyncEvent() : ready_(false) {
}

void SyncEvent::wait() {
  boost::unique_lock<boost::mutex> lock(mu_);
  while (!ready_) {
    cond_.wait(lock);
  }
}

void SyncEvent::signal() {
  boost::unique_lock<boost::mutex> lock(mu_);
  ready_ = true;
  cond_.notify_all();
}

}  // namespace core
}  // namespace artm
