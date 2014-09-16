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

#ifndef RPCZ_FILE_GENERATOR_H
#define RPCZ_FILE_GENERATOR_H

#include <string>
#include <vector>

#include "rpcz/plugin/cpp/rpcz_cpp_service.h"

namespace rpcz {
namespace plugin {
namespace cpp {
class ServiceGenerator;
}  // namespace cpp
}  // namespace plugin
}  // namespace rpcz

namespace google {
namespace protobuf {
class FileDescriptor;
class ServiceDescriptor;

namespace io {
class Printer;
}
}
}

namespace rpcz {
namespace plugin {
namespace cpp {

class FileGenerator {
  public:
    FileGenerator(const google::protobuf::FileDescriptor* file,
                  const std::string& dllexport_decl);

    ~FileGenerator();

    void GenerateHeader(google::protobuf::io::Printer* printer);

    void GenerateSource(google::protobuf::io::Printer* printer);

  private:
    void GenerateNamespaceOpeners(google::protobuf::io::Printer* printer);

    void GenerateNamespaceClosers(google::protobuf::io::Printer* printer);

    void GenerateBuildDescriptors(google::protobuf::io::Printer* printer);

    std::vector<std::string> package_parts_;
    std::vector<ServiceGenerator*> service_generators_;
    const ::google::protobuf::FileDescriptor* file_;
    std::string dllexport_decl_;
};

}  // namespace cpp
}  // namespace plugin
}  // namespace rpcz
#endif
