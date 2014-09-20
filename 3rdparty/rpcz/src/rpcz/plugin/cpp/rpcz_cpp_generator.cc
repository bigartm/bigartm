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

#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <string>
#include <utility>
#include <vector>

#include "rpcz/plugin/cpp/file_generator.h"
#include "rpcz/plugin/cpp/rpcz_cpp_generator.h"

namespace rpcz {
namespace plugin {
namespace cpp {

using std::pair;
using std::string;
using std::vector;

RpczCppGenerator::RpczCppGenerator() {}
RpczCppGenerator::~RpczCppGenerator() {}

namespace {
inline bool HasSuffixString(const string& str,
                            const string& suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline string StripSuffixString(const string& str, const string& suffix) {
  if (HasSuffixString(str, suffix)) {
    return str.substr(0, str.size() - suffix.size());
  } else {
    return str;
  }
}
}

bool RpczCppGenerator::Generate(
    const ::google::protobuf::FileDescriptor* file,
    const string& parameter,
    ::google::protobuf::compiler::GeneratorContext* generator_context,
    string* error) const {
  vector<pair<string, string> > options;
  ::google::protobuf::compiler::ParseGeneratorParameter(parameter, &options);

  // If the dllexport_decl option is passed to the compiler, we need to write
  // it in front of every symbol that should be exported if this .proto is
  // compiled into a Windows DLL.  E.g., if the user invokes the protocol
  // compiler as:
  //   protoc --cpp_out=dllexport_decl=FOO_EXPORT:outdir foo.proto
  // then we'll define classes like this:
  //   class FOO_EXPORT Foo {
  //     ...
  //   }
  // FOO_EXPORT is a macro which should expand to __declspec(dllexport) or
  // __declspec(dllimport) depending on what is being compiled.
  string dllexport_decl;

  for (size_t i = 0; i < options.size(); i++) {
    if (options[i].first == "dllexport_decl") {
      dllexport_decl = options[i].second;
    } else {
      *error = "Unknown generator option: " + options[i].first;
      return false;
    }
  }

  // -----------------------------------------------------------------


  string basename = StripSuffixString(file->name(), ".proto");
  basename.append(".rpcz");

  FileGenerator file_generator(file, dllexport_decl);

  // Generate header.
  {
    ::google::protobuf::scoped_ptr<google::protobuf::io::ZeroCopyOutputStream> output(
        generator_context->Open(basename + ".h"));
    ::google::protobuf::io::Printer printer(output.get(), '$');
    file_generator.GenerateHeader(&printer);
  }

  // Generate cc file.
  {
    ::google::protobuf::scoped_ptr<google::protobuf::io::ZeroCopyOutputStream> output(
        generator_context->Open(basename + ".cc"));
    ::google::protobuf::io::Printer printer(output.get(), '$');
    file_generator.GenerateSource(&printer);
  }

  return true;
}
}  // namespace cpp
}  // namespace plugin
}  // namespace rpcz
