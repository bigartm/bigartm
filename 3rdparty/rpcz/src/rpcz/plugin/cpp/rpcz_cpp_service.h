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

#ifndef RPCZ_CPP_SERVICE_H
#define RPCZ_CPP_SERVICE_H

#include <google/protobuf/descriptor.h>
#include <google/protobuf/stubs/common.h>
#include <map>
#include <string>

namespace google {
namespace protobuf {
class ServiceDescriptor;
}  // namespace protobuf
}  // namespace google

namespace google {
namespace protobuf {
  namespace io {
    class Printer;             // printer.h
  }
}
}

namespace rpcz {
namespace plugin {
namespace cpp {

class ServiceGenerator {
 public:
  // See generator.cc for the meaning of dllexport_decl.
  explicit ServiceGenerator(
      const google::protobuf::ServiceDescriptor* descriptor,
      const std::string& dllexport_decl);
  ~ServiceGenerator();

  // Header stuff.

  // Generate the class definitions for the service's interface and the
  // stub implementation.
  void GenerateDeclarations(google::protobuf::io::Printer* printer);

  // Source file stuff.

  // Generate code that initializes the global variable storing the service's
  // descriptor.
  void GenerateDescriptorInitializer(google::protobuf::io::Printer* printer,
                                     int index);

  // Generate implementations of everything declared by GenerateDeclarations().
  void GenerateImplementation(google::protobuf::io::Printer* printer);

 private:
  enum RequestOrResponse { REQUEST, RESPONSE };
  enum VirtualOrNon { VIRTUAL, NON_VIRTUAL };

  // Header stuff.

  // Generate the service abstract interface.
  void GenerateInterface(google::protobuf::io::Printer* printer);

  // Generate the stub class definition.
  void GenerateStubDefinition(google::protobuf::io::Printer* printer);

  // Prints signatures for all methods in the
  void GenerateMethodSignatures(VirtualOrNon virtual_or_non,
                                google::protobuf::io::Printer* printer,
                                bool stub);

  // Source file stuff.

  // Generate the default implementations of the service methods, which
  // produce a "not implemented" error.
  void GenerateNotImplementedMethods(google::protobuf::io::Printer* printer);

  // Generate the CallMethod() method of the service.
  void GenerateCallMethod(google::protobuf::io::Printer* printer);

  // Generate the Get{Request,Response}Prototype() methods.
  void GenerateGetPrototype(RequestOrResponse which,
                            google::protobuf::io::Printer* printer);

  // Generate the stub's implementations of the service methods.
  void GenerateStubMethods(google::protobuf::io::Printer* printer);

  const google::protobuf::ServiceDescriptor* descriptor_;
  std::map<std::string, std::string> vars_;

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(ServiceGenerator);
};

}  // namespace cpp
}  // namespace plugin
}  // namespace rpcz
#endif  // RPCZ_CPP_SERVICE_H
