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

#include <iostream>
#include <vector>
#include <boost/bind.hpp>
#include <boost/program_options.hpp>
#include <google/protobuf/compiler/importer.h>
#include <google/protobuf/dynamic_message.h>
#include <google/protobuf/text_format.h>
#include "rpcz/application.hpp"
#include "rpcz/rpc_channel.hpp"
#include "rpcz/service.hpp"
#include "rpcz/rpc.hpp"

using google::protobuf::DynamicMessageFactory;
using google::protobuf::FileDescriptor;
using google::protobuf::Message;
using google::protobuf::MethodDescriptor;
using google::protobuf::ServiceDescriptor;
using google::protobuf::ShutdownProtobufLibrary;
using google::protobuf::TextFormat;
using google::protobuf::compiler::DiskSourceTree;
using google::protobuf::compiler::Importer;
using google::protobuf::compiler::MultiFileErrorCollector;
using std::endl;
using std::cout;
using std::cerr;

namespace po = boost::program_options;

std::string FLAGS_proto;
std::string FLAGS_service_name;
std::vector<std::string> FLAGS_proto_path;

static const char *PNAME = "zsendrpc";

namespace rpcz {
class ErrorCollector : public MultiFileErrorCollector {
  void AddError(
      const std::string& filename, int line, int /* column */ ,
      const std::string& message) {
    cerr << filename << ":" << line << ":" << message << endl;
  }
};

int run_call(const std::string& endpoint,
             const std::string& method,
             const std::string& payload) {
  DiskSourceTree disk_source_tree;
  ErrorCollector error_collector;
  for_each(FLAGS_proto_path.begin(), FLAGS_proto_path.end(),
      boost::bind(&DiskSourceTree::MapPath,
          &disk_source_tree, _1, _1));
  Importer imp(&disk_source_tree, &error_collector);

  const FileDescriptor* file_desc = imp.Import(FLAGS_proto);

  if (file_desc == NULL) {
    cerr << "Could not load proto '" << FLAGS_proto
         << "'" << endl;
    return -1;
  }
  if (method.find('.') == method.npos) {
    cerr << "<service.method> must contain a dot: '" << method << "'"
              << endl;
    return -1;
  }
  std::string service_name(method, 0, method.find_last_of('.'));
  std::string method_name(method, method.find_last_of('.') + 1);
  const ::ServiceDescriptor* service_desc =
      file_desc->FindServiceByName(service_name);
  if (service_desc == NULL) {
    cerr << "Could not find service '" << service_name
              << "' in proto definition." << endl;
    return -1;
  }
  const ::MethodDescriptor* method_desc =
      service_desc->FindMethodByName(method_name);
  if (method_desc == NULL) {
    cerr << "Could not find method '" << method_name
              << "' in proto definition (but service was found)." << endl;
    return -1;
  }

  DynamicMessageFactory factory;
  Message *request = factory.GetPrototype(
      method_desc->input_type())->New();
  if (request == NULL) {
    cerr << "Could not allocate request.";
    return -1;
  }
  if (!TextFormat::ParseFromString(payload, request)) {
    cerr << "Could not parse the given ASCII message." << endl;
    return -1;
  }

  application app;
  scoped_ptr<rpc_channel> channel(app.create_rpc_channel(endpoint));
  rpc rpc;
  ::Message *reply = factory.GetPrototype(
      method_desc->output_type())->New();
  channel->call_method(
      FLAGS_service_name.empty() ? service_name : FLAGS_service_name,
      method_desc, *request, reply, &rpc, NULL);
  rpc.wait();

  if (rpc.get_status() != status::OK) {
    cerr << "status: " << rpc.get_status() << endl;
    cerr << "Error " << rpc.get_application_error_code() << ": "
        << rpc.get_error_message() << endl;
  } else {
    std::string out;
    ::TextFormat::PrintToString(*reply, &out);
    cout << out << endl;
  }
  delete request;
  delete reply;
  return 0;
}

#define ARGV_ERROR -2

int run(std::vector<std::string> args) {
  if (args.empty()) {
    cerr << "Expecting a command." << endl;
    return ARGV_ERROR;
  }
  std::string command(args[0]);
  if (command != "call") {
    cerr << "Only the call command is supported" << endl;
    return ARGV_ERROR;
  } else {
    if (args.size() != 4) {
      cerr << "call needs 3 arguments:" <<
          "call <endpoint> <service.method> <payload>" 
          << endl << endl;
      return ARGV_ERROR;
    }
    std::string endpoint(args[1]);
    std::string method(args[2]);
    std::string payload(args[3]);
    return run_call(endpoint, method, payload);
  }
  return 0;
}
}  // namespace rpcz

void show_usage(const char* pname, const po::options_description& desc) {
  cout << pname << " Usage Instructions" << endl
       << endl
       << pname << " --proto=file.proto <command> [args]" << endl
       << endl
       << "Where <command> is one of the following: " << endl
       << "  call" << endl
       << endl;
  // cout << desc;
}

int main(int argc, char *argv[]) {
  po::options_description desc("Allowed options");
  po::variables_map vm;

  desc.add_options()
      ("help", "produce help message")
      ("proto", po::value<std::string>(&FLAGS_proto)->required(),
       "Protocol Buffer file to use.")
      ("proto_path", po::value<std::vector<std::string> >(&FLAGS_proto_path),
       "List of directories to search.")
      ("service_name", po::value<std::string>(&FLAGS_service_name),
       "service name to use. Leave empty to use the same service name as in "
       "the proto definition.");

  po::positional_options_description p;
  po::parsed_options parsed = po::command_line_parser(argc, argv).
        options(desc).allow_unregistered().run();

  try {
    po::store(parsed, vm);
  } catch (po::error &e) {
    cerr << "Command line error: " << e.what() << endl;
    show_usage(PNAME, desc);
    return 1;
  }

  if (vm.count("help")) {
    show_usage(PNAME, desc);
    return 1;
  }

  try {
    po::notify(vm);
  } catch (po::error &e) {
    cerr << "Command line error: " << e.what() << endl;
    show_usage(PNAME, desc);
    return 1;
  }
  std::vector<std::string> positional = po::collect_unrecognized(
      parsed.options, po::include_positional);
  int retval = rpcz::run(positional);
  if (retval == ARGV_ERROR) {
    retval = 1;
    show_usage(PNAME, desc);
  }
  return retval;
}
