// Copyright 2014, Additive Regularization of Topic Models.

#include <memory>

#include "boost/thread.hpp"
#include "gtest/gtest.h"

#include "rpcz/application.hpp"
#include "rpcz/server.hpp"
#include "rpcz/rpc.hpp"

#include "artm/core/zmq_context.h"

#include "artm_tests/rpcz_canary_service.pb.h"
#include "artm_tests/rpcz_canary_service.rpcz.h"

// To generate protobuf files:
// copy utils/protoc-2.5.0-win32/protoc.exe to /src folder
// cd to /src folder and execute the following:
// .\protoc.exe --cpp_out=. --rpcz_plugin_out=. .\artm_tests\rpcz_canary_service.proto

const std::string error_message = "Some error had happened!";
const int error_code = -999;

class SearchServiceImpl : public SearchService {
  virtual void Search(
      const SearchRequest& request,
      rpcz::reply<SearchResponse> reply) {
    if (request.query() == "make_error") {
      reply.Error(error_code, error_message);
    } else {
      SearchResponse response;
      response.add_results("result1 for " + request.query());
      response.add_results("this is result2");
      reply.send(response);
    }
  }
};

std::unique_ptr<rpcz::application> server_application;

void RpczServerThreadFunction() {
  SearchServiceImpl search_service;
  rpcz::application::options options(3);
  options.zeromq_context = ::artm::core::ZmqContext::singleton().get();
  server_application.reset(new rpcz::application(options));
  rpcz::server server(*server_application);
  server.register_service(&search_service);
  server.bind("tcp://*:5555");
  server_application->run();
}

void ConnectAndQuery(int timeout = -1L, bool make_error = false) {
  rpcz::application::options options(3);
  options.zeromq_context = ::artm::core::ZmqContext::singleton().get();
  rpcz::application client_application(options);

  SearchService_Stub search_service_proxy(
    client_application.create_rpc_channel("tcp://localhost:5555"), true);

  SearchRequest request;
  SearchResponse response;

  try {
    request.set_page_number(10);
    if (make_error) {
      request.set_query("make_error");
    } else {
      request.set_query("my query");
    }
    search_service_proxy.Search(request, &response, timeout);

    ASSERT_EQ(response.results_size(), 2);
    ASSERT_EQ(response.results(0), "result1 for my query");
    ASSERT_EQ(response.results(1), "this is result2");
  } catch(const ::rpcz::rpc_error& error) {
    if (timeout != -1L) {
      ASSERT_EQ(error.get_status(), rpcz::status::DEADLINE_EXCEEDED);
    } else {
      ASSERT_TRUE(make_error);
      if (make_error) {
        ASSERT_EQ(error.get_application_error_code(), error_code);
        ASSERT_EQ(error.get_error_message(), error_message);
      }
    }
  }
}

// artm_tests.exe --gtest_filter=Rpcz.Canary
TEST(Rpcz, Canary) {
  boost::thread t(&RpczServerThreadFunction);

  ConnectAndQuery();

  ASSERT_TRUE(server_application != nullptr);
  server_application->terminate();
  t.join();

  server_application.reset();
}

// artm_tests.exe --gtest_filter=Rpcz.Timeout
TEST(Rpcz, Timeout) {
  // Timeout test does not create a server to connect to, and sets the timeout for the first RPCZ call.
  // The expectation of the test is to get an exception with rpcz::status::DEADLINE_EXCEEDED being set.
  ConnectAndQuery(10);
}

// artm_tests.exe --gtest_filter=Rpcz.ErrorHandling
TEST(Rpcz, ErrorHandling) {
  boost::thread t(&RpczServerThreadFunction);

  ConnectAndQuery(-1L, true);

  ASSERT_TRUE(server_application != nullptr);
  server_application->terminate();
  t.join();

  server_application.reset();
}
