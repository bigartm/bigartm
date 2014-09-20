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

// Author: robinson@google.com (Will Robinson)
//
// Generates Python code for a given .proto file.

#ifndef RPCZ_PLUGIN_PYTHON_RPCZ_PYTHON_GENERATOR_H
#define RPCZ_PLUGIN_PYTHON_RPCZ_PYTHON_GENERATOR_H

#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/stubs/common.h>
#include <string>

namespace google {
namespace protobuf {
class Descriptor;
class EnumDescriptor;
class EnumValueDescriptor;
class FieldDescriptor;
class ServiceDescriptor;
class FileDescriptor;
namespace io {
class Printer;
}  // namespace io
}  // namespace protobuf
}  // namespace google

namespace rpcz {
namespace plugin {
namespace python {

// CodeGenerator implementation for generated Python protocol buffer classes.
// If you create your own protocol compiler binary and you want it to support
// Python output, you can do so by registering an instance of this
// CodeGenerator with the CommandLineInterface in your main() function.
class LIBPROTOC_EXPORT Generator
    : public google::protobuf::compiler::CodeGenerator {
 public:
  Generator();
  virtual ~Generator();

  // CodeGenerator methods.
  virtual bool Generate(
      const google::protobuf::FileDescriptor* file,
      const std::string& parameter,
      google::protobuf::compiler::GeneratorContext* generator_context,
      std::string* error) const;

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(Generator);
};

class FileGenerator {
 public:
  FileGenerator(const google::protobuf::FileDescriptor* file,
                google::protobuf::io::Printer* printer);

  bool Run();

 private:
  void PrintImports() const;

  void PrintServices() const;
  void PrintServiceDescriptor(const google::protobuf::ServiceDescriptor& descriptor) const;
  void PrintServiceClass(const google::protobuf::ServiceDescriptor& descriptor) const;
  void PrintServiceStub(const google::protobuf::ServiceDescriptor& descriptor) const;

  std::string OptionsValue(const std::string& class_name,
                      const std::string& serialized_options) const;
  bool GeneratingDescriptorProto() const;

  template <typename DescriptorT>
  std::string ModuleLevelDescriptorName(const DescriptorT& descriptor) const;
  std::string ModuleLevelMessageName(const google::protobuf::Descriptor& descriptor) const;
  std::string ModuleLevelServiceDescriptorName(
      const google::protobuf::ServiceDescriptor& descriptor) const;

  template <typename DescriptorT, typename DescriptorProtoT>
  void PrintSerializedPbInterval(
      const DescriptorT& descriptor, DescriptorProtoT& proto) const;

  const google::protobuf::FileDescriptor* file_;
  google::protobuf::io::Printer* printer_;
  std::string file_descriptor_serialized_;

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(FileGenerator);
};
}  // namespace python
}  // namespace plugin
}  // namespace rpcz
#endif  // RPCZ_PLUGIN_PYTHON_RPCZ_PYTHON_GENERATOR_H
