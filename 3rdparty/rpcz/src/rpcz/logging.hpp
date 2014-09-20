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

#ifndef RPCZ_LOGGING_H
#define RPCZ_LOGGING_H

// Internal logging and checks. We piggyback on the facilities that come
// from the protobuf library.
#include "google/protobuf/stubs/common.h"

// CHECK_* functions cause the program to terminate. They should be used only
// for internal consistency checks.
#define CHECK_GE GOOGLE_CHECK_GE
#define CHECK_NE GOOGLE_CHECK_NE
#define CHECK_EQ GOOGLE_CHECK_EQ
#define CHECK GOOGLE_CHECK
#define CHECK_NOTNULL(val) \
      ::rpcz::internal::CheckNotNull( \
          __FILE__, __LINE__, "'" #val "' Must be non NULL", (val))

// Logs a debugging message to stderr. Does not have eny effect when compiling
// with -DNDEBUG
#define LOG GOOGLE_LOG

#ifndef NDEBUG
  #define DLOG GOOGLE_DLOG
#else
  #define DLOG true ? (void)0 : LOG
#endif

namespace rpcz {
namespace internal {
// A small helper for CHECK_NOTNULL().
template <typename T>
T* CheckNotNull(const char *file, int line, const char *names, T* t) {
  GOOGLE_CHECK(t != NULL) << names;
  return t;
}
}  // namespace internal
}  // namespace rpcz

#endif
