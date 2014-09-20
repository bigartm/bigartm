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

#include "zmq.hpp"
#include "rpcz/application.hpp"
#include "rpcz/macros.hpp"
#include "gtest/gtest.h"

namespace rpcz {

class application_test : public ::testing::Test {
 protected:
  void InitDefaultApp() {
    application_.reset(new application);
  }

  scoped_ptr<application> application_;
};

TEST_F(application_test, Initializes) {
  InitDefaultApp();
}

TEST_F(application_test, InitializesWithProvidedZeroMQContext) {
  zmq::context_t* context = new zmq::context_t(1);
  application::options options;
  options.zeromq_context = context;
  application_.reset(new application(options));
  application_.reset();
  delete context;
}

}  // namespace rpcz
