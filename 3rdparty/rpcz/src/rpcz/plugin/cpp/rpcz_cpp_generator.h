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

#ifndef RPCZ_RPCZ_CPP_GENERATOR_H
#define RPCZ_RPCZ_CPP_GENERATOR_H

#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/stubs/common.h>
#include <string>

namespace google {
namespace protobuf {
class FileDescriptor;
}  // namespace protobuf
}  // namespace google

namespace rpcz {
namespace plugin {
namespace cpp {

class LIBPROTOC_EXPORT RpczCppGenerator :
    public ::google::protobuf::compiler::CodeGenerator {
 public:
  RpczCppGenerator();
  ~RpczCppGenerator();

  bool Generate(
      const ::google::protobuf::FileDescriptor* file,
      const ::std::string& parameter,
      ::google::protobuf::compiler::GeneratorContext* generator_context,
      std::string* error) const;

 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(RpczCppGenerator);
};
}  // namespace cpp
}  // namespace plugin
}  // namespace rpcz
#endif
