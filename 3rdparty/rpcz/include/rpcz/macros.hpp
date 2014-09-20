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

#ifndef RPCZ_BASE_H
#define RPCZ_BASE_H

#include "google/protobuf/stubs/common.h"

namespace rpcz {
using google::protobuf::scoped_ptr; 
using google::protobuf::uint64;
using google::protobuf::int64;

// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&);               \
    void operator=(const TypeName&)

// Deletes each pointer in the range [begin, end)
template<typename IteratorType>
void delete_container_pointers(const IteratorType& begin,
                             const IteratorType& end) {
  for (IteratorType i = begin; i != end; ++i) {
    delete *i;
  }
}

// For each item in the range [begin, end), delete item->first and item->second.
template<typename IteratorType>
void delete_container_pair_pointers(const IteratorType& begin,
                                 const IteratorType& end) {
  for (IteratorType i = begin; i != end; ++i) {
    delete i->first;
    delete i->second;
  }
}

// For each item in the range [begin, end), delete item->second.
template<typename IteratorType>
void delete_container_second_pointer(const IteratorType& begin,
                                  const IteratorType& end) {
  for (IteratorType i = begin; i != end; ++i) {
    delete i->second;
  }
}
}  // namespace rpcz
#endif
