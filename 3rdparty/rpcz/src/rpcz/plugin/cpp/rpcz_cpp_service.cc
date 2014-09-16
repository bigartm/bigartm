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

#include "rpcz/plugin/cpp/rpcz_cpp_service.h"

#include <google/protobuf/descriptor.h>
#include <google/protobuf/io/printer.h>

#include "rpcz/plugin/common/strutil.h"
#include "rpcz/plugin/cpp/cpp_helpers.h"

namespace rpcz {
namespace plugin {
namespace cpp {

using namespace google::protobuf;
using namespace google::protobuf::compiler::cpp;

ServiceGenerator::ServiceGenerator(const ServiceDescriptor* descriptor,
                                   const string& dllexport_decl)
  : descriptor_(descriptor) {
  vars_["classname"] = descriptor_->name();
  vars_["full_name"] = descriptor_->full_name();
  if (dllexport_decl.empty()) {
    vars_["dllexport"] = "";
  } else {
    vars_["dllexport"] = dllexport_decl + " ";
  }
}

ServiceGenerator::~ServiceGenerator() {}

void ServiceGenerator::GenerateDeclarations(io::Printer* printer) {
  // Forward-declare the stub type.
  printer->Print(vars_,
    "class $classname$_Stub;\n"
    "\n");

  GenerateInterface(printer);
  GenerateStubDefinition(printer);
}

void ServiceGenerator::GenerateInterface(io::Printer* printer) {
  printer->Print(vars_,
    "class $dllexport$$classname$ : public rpcz::service {\n"
    " protected:\n"
    "  // This class should be treated as an abstract interface.\n"
    "  inline $classname$() {};\n"
    " public:\n"
    "  virtual ~$classname$();\n");
  printer->Indent();

  printer->Print(vars_,
    "\n"
    "typedef $classname$_Stub Stub;\n"
    "\n"
    "static const ::google::protobuf::ServiceDescriptor* descriptor();\n"
    "\n");

  GenerateMethodSignatures(VIRTUAL, printer, false);

  printer->Print(
    "\n"
    "// implements Service ----------------------------------------------\n"
    "\n"
    "const ::google::protobuf::ServiceDescriptor* GetDescriptor();\n"
    "void call_method(const ::google::protobuf::MethodDescriptor* method,\n"
    "                const ::google::protobuf::Message& request,\n"
    "                ::rpcz::server_channel* channel);\n"
    "const ::google::protobuf::Message& GetRequestPrototype(\n"
    "  const ::google::protobuf::MethodDescriptor* method) const;\n"
    "const ::google::protobuf::Message& GetResponsePrototype(\n"
    "  const ::google::protobuf::MethodDescriptor* method) const;\n");

  printer->Outdent();
  printer->Print(vars_,
    "\n"
    " private:\n"
    "  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS($classname$);\n"
    "};\n"
    "\n");
}

void ServiceGenerator::GenerateStubDefinition(io::Printer* printer) {
  printer->Print(vars_,
    "class $dllexport$$classname$_Stub {\n"
    " public:\n");

  printer->Indent();

  printer->Print(vars_,
    "$classname$_Stub(::rpcz::rpc_channel* channel, \n"
    "                 bool owns_channel=false);\n"
    "$classname$_Stub(::rpcz::rpc_channel* channel, \n"
    "                 const ::std::string& service_name,\n"
    "                 bool owns_channel=false);\n"
    "~$classname$_Stub();\n"
    "\n"
    "inline ::rpcz::rpc_channel* channel() { return channel_; }\n"
    "\n"
    "\n");

  GenerateMethodSignatures(NON_VIRTUAL, printer, true);

  printer->Outdent();
  printer->Print(vars_,
    " private:\n"
    "  ::rpcz::rpc_channel* channel_;\n"
    "  ::std::string service_name_;\n"
    "  bool owns_channel_;\n"
    "  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS($classname$_Stub);\n"
    "};\n"
    "\n");
}

void ServiceGenerator::GenerateMethodSignatures(
    VirtualOrNon virtual_or_non, io::Printer* printer, bool stub) {
  for (int i = 0; i < descriptor_->method_count(); i++) {
    const MethodDescriptor* method = descriptor_->method(i);
    map<string, string> sub_vars;
    sub_vars["name"] = method->name();
    sub_vars["input_type"] = ClassName(method->input_type(), true);
    sub_vars["output_type"] = ClassName(method->output_type(), true);
    sub_vars["virtual"] = virtual_or_non == VIRTUAL ? "virtual " : "";

    if (stub) {
      printer->Print(sub_vars,
                     "$virtual$void $name$(const $input_type$& request,\n"
                     "                     $output_type$* response,\n"
                     "                     ::rpcz::rpc* rpc,"
                     "                     ::rpcz::closure* done);\n");
      printer->Print(sub_vars,
                     "$virtual$void $name$(const $input_type$& request,\n"
                     "                     $output_type$* response,\n"
                     "                     long deadline_ms = -1);\n");
    } else {
      printer->Print(
          sub_vars,
          "$virtual$void $name$(const $input_type$& request,\n"
          "                     ::rpcz::reply< $output_type$> response);\n");
    }
  }
}

// ===================================================================

void ServiceGenerator::GenerateDescriptorInitializer(
    io::Printer* printer, int index) {
  map<string, string> vars;
  vars["classname"] = descriptor_->name();
  vars["index"] = SimpleItoa(index);

  printer->Print(vars,
    "$classname$_descriptor_ = file->service($index$);\n");
}

// ===================================================================

void ServiceGenerator::GenerateImplementation(io::Printer* printer) {
  printer->Print(vars_,
    "$classname$::~$classname$() {}\n"
    "\n"
    "const ::google::protobuf::ServiceDescriptor* $classname$::descriptor() {\n"
    "  protobuf_AssignDescriptorsOnce();\n"
    "  return $classname$_descriptor_;\n"
    "}\n"
    "\n"
    "const ::google::protobuf::ServiceDescriptor* $classname$::GetDescriptor() {\n"
    "  protobuf_AssignDescriptorsOnce();\n"
    "  return $classname$_descriptor_;\n"
    "}\n"
    "\n");

  // Generate methods of the interface.
  GenerateNotImplementedMethods(printer);
  GenerateCallMethod(printer);
  GenerateGetPrototype(REQUEST, printer);
  GenerateGetPrototype(RESPONSE, printer);

  // Generate stub implementation.
  printer->Print(vars_,
    "$classname$_Stub::$classname$_Stub(::rpcz::rpc_channel* channel,\n"
    "                                   const ::std::string& service_name,\n"
    "                                   bool owns_channel)\n"
    "  : channel_(channel), service_name_(service_name),\n"
    "    owns_channel_(owns_channel) {}\n"
    "$classname$_Stub::$classname$_Stub(::rpcz::rpc_channel* channel,\n"
    "                                   bool owns_channel)\n"
    "  : channel_(channel), service_name_($classname$::descriptor()->name()),\n"
    "    owns_channel_(owns_channel) {}\n"
    "$classname$_Stub::~$classname$_Stub() {\n"
    "  if (owns_channel_) delete channel_;\n"
    "}\n"
    "\n");

  GenerateStubMethods(printer);
}

void ServiceGenerator::GenerateNotImplementedMethods(io::Printer* printer) {
  for (int i = 0; i < descriptor_->method_count(); i++) {
    const MethodDescriptor* method = descriptor_->method(i);
    map<string, string> sub_vars;
    sub_vars["classname"] = descriptor_->name();
    sub_vars["name"] = method->name();
    sub_vars["index"] = SimpleItoa(i);
    sub_vars["input_type"] = ClassName(method->input_type(), true);
    sub_vars["output_type"] = ClassName(method->output_type(), true);

    printer->Print(sub_vars,
      "void $classname$::$name$(const $input_type$&,\n"
      "                         ::rpcz::reply< $output_type$> reply) {\n"
      "  reply.Error(::rpcz::application_error::METHOD_NOT_IMPLEMENTED,\n"
      "              \"Method $name$() not implemented.\");\n"
      "}\n"
      "\n");
  }
}

void ServiceGenerator::GenerateCallMethod(io::Printer* printer) {
  printer->Print(vars_,
    "void $classname$::call_method(const ::google::protobuf::MethodDescriptor* method,\n"
    "                             const ::google::protobuf::Message& request,\n"
    "                             ::rpcz::server_channel* channel) {\n"
    "  GOOGLE_DCHECK_EQ(method->service(), $classname$_descriptor_);\n"
    "  switch(method->index()) {\n");

  for (int i = 0; i < descriptor_->method_count(); i++) {
    const MethodDescriptor* method = descriptor_->method(i);
    map<string, string> sub_vars;
    sub_vars["name"] = method->name();
    sub_vars["index"] = SimpleItoa(i);
    sub_vars["input_type"] = ClassName(method->input_type(), true);
    sub_vars["output_type"] = ClassName(method->output_type(), true);

    // Note:  down_cast does not work here because it only works on pointers,
    //   not references.
    printer->Print(sub_vars,
      "    case $index$:\n"
      "      $name$(\n"
      "          *::google::protobuf::down_cast<const $input_type$*>(&request),\n"
      "          ::rpcz::reply< $output_type$>(channel));\n"
      "      break;\n");
  }

  printer->Print(vars_,
    "    default:\n"
    "      GOOGLE_LOG(FATAL) << \"Bad method index; this should never happen.\";\n"
    "      break;\n"
    "  }\n"
    "}\n"
    "\n");
}

void ServiceGenerator::GenerateGetPrototype(RequestOrResponse which,
                                            io::Printer* printer) {
  if (which == REQUEST) {
    printer->Print(vars_,
      "const ::google::protobuf::Message& $classname$::GetRequestPrototype(\n");
  } else {
    printer->Print(vars_,
      "const ::google::protobuf::Message& $classname$::GetResponsePrototype(\n");
  }

  printer->Print(vars_,
    "    const ::google::protobuf::MethodDescriptor* method) const {\n"
    "  GOOGLE_DCHECK_EQ(method->service(), descriptor());\n"
    "  switch(method->index()) {\n");

  for (int i = 0; i < descriptor_->method_count(); i++) {
    const MethodDescriptor* method = descriptor_->method(i);
    const Descriptor* type =
      (which == REQUEST) ? method->input_type() : method->output_type();

    map<string, string> sub_vars;
    sub_vars["index"] = SimpleItoa(i);
    sub_vars["type"] = ClassName(type, true);

    printer->Print(sub_vars,
      "    case $index$:\n"
      "      return $type$::default_instance();\n");
  }

  printer->Print(vars_,
    "    default:\n"
    "      GOOGLE_LOG(FATAL) << \"Bad method index; this should never happen.\";\n"
    "      return *reinterpret_cast< ::google::protobuf::Message*>(NULL);\n"
    "  }\n"
    "}\n"
    "\n");
}

void ServiceGenerator::GenerateStubMethods(io::Printer* printer) {
  for (int i = 0; i < descriptor_->method_count(); i++) {
    const MethodDescriptor* method = descriptor_->method(i);
    map<string, string> sub_vars;
    sub_vars["classname"] = descriptor_->name();
    sub_vars["name"] = method->name();
    sub_vars["index"] = SimpleItoa(i);
    sub_vars["input_type"] = ClassName(method->input_type(), true);
    sub_vars["output_type"] = ClassName(method->output_type(), true);

    printer->Print(sub_vars,
      "void $classname$_Stub::$name$(const $input_type$& request,\n"
      "                              $output_type$* response,\n"
      "                              ::rpcz::rpc* rpc,\n"
      "                              ::rpcz::closure* done) {\n"
      "  channel_->call_method(service_name_,\n"
      "                        $classname$::descriptor()->method($index$),\n"
      "                        request, response, rpc, done);\n"
      "}\n");
    printer->Print(sub_vars,
      "void $classname$_Stub::$name$(const $input_type$& request,\n"
      "                              $output_type$* response,\n"
      "                              long deadline_ms) {\n"
      "  ::rpcz::rpc rpc;\n"
      "  rpc.set_deadline_ms(deadline_ms);\n"
      "  channel_->call_method(service_name_,\n"
      "                        $classname$::descriptor()->method($index$),\n"
      "                        request, response, &rpc, NULL);\n"
      "  rpc.wait();\n"
      "  if (!rpc.ok()) {\n"
      "    throw ::rpcz::rpc_error(rpc);\n"
      "  }\n"
      "}\n");
  }
}

}  // namespace cpp
}  // namespace plugin
}  // namespace rpcz
