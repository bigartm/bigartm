#include <iostream>
#include <string>

#include "artm/cpp_interface.h"
#include "artm/messages.pb.h"

int main(int argc, char * argv[]) {
  if (argc < 2) {
    std::cout << "Usage:\n\t./node_controller <endpoint> [GLOG_switches]\n\n";
    std::cout << "Examples:\n";
    std::cout << "\t./node_controller tcp://*:5555\n";
    std::cout << "\t./node_controller tcp://*:5555 --logtostderr=1\n";
    std::cout << "\tset GLOG_logtostderr=1 & ./node_controller tcp://*:5555\n\n";
    std::cout << "To connect to the node_controller replace '*' in the endpoint with\n";
    std::cout << "IP address or DNS name of the host running the node_controller.\n";
    std::cout << "For configuration of GLOG_switches please refer to\n";
    std::cout << "http://google-glog.googlecode.com/svn/trunk/doc/glog.html\n\n";

    return 0;
  }

  // Logging to std::cerr is done intentionally.
  // A common scenario is to run node_controller with custom GLOG settings, for example:
  //  set GLOG_logtostderr=1 & node_controller.exe tcp://*:5555
  // In this case logging to std::cout would cause badly overlapped messages with GLOG logging 
  // to std::cerr.
  std::cerr << "Starting NodeController at " << argv[1] << "...\n";
  ::artm::NodeControllerConfig node_config;
  node_config.set_create_endpoint(argv[1]);
  ::artm::NodeController node_controller(node_config);

  std::cerr << "NodeController is now running. Type Ctrl+Z or Ctrl+C to quit.\n";
  while(true) {
    std::string str;
    std::cin >> str;
  }

  return 0;
}
