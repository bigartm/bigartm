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

#include "rpcz/callback.hpp"

#include <string>
#include "gtest/gtest.h"

namespace rpcz {

static bool called = false;
void ZeroCallback() {
  called = true;
}

class test_object {
 public:
  void Method(int arg1, std::string arg2) {
    EXPECT_EQ(17, arg1);
    EXPECT_EQ("super!", arg2);
    called = true;
  }
};

TEST(CallbackTest, TestCallbackSimple) {
  closure* c = new_callback(&ZeroCallback);
  called = false;
  c->run();
  EXPECT_TRUE(called);
}

TEST(CallbackTest, TestPermanentCallbackSimple) {
  closure* c = new_permanent_callback(&ZeroCallback);
  for (int i = 1; i < 10; ++i) {
    called = false;
    c->run();
    EXPECT_TRUE(called);
  }
  delete c;
  EXPECT_TRUE(called);
}

TEST(CallbackTest, TestMethodCallback) {
  test_object object;
  called = false;
  closure *c = new_callback(&object, &test_object::Method, 17,
                           std::string("super!"));
  c->run();
  EXPECT_TRUE(called);
}

TEST(CallbackTest, TestMethodPermanentCallback) {
  test_object object;
  called = false;
  closure *c = new_permanent_callback(&object, &test_object::Method, 17,
                                      std::string("super!"));
  for (int i = 1; i < 10; ++i) {
    called = false;
    c->run();
    EXPECT_TRUE(called);
  }
  delete c;
  EXPECT_TRUE(called);
}
}  // namespace rpcz
