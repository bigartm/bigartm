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
#include "rpcz/rpcz.hpp"
#include "cpp/search.pb.h"
#include "cpp/search.rpcz.h"

using namespace std;

namespace examples {

class SearchServiceImpl : public SearchService {

  virtual void Search(
      const SearchRequest& request,
      rpcz::reply<SearchResponse> reply) {
    cout << "Got request for '" << request.query() << "'" << endl;
    SearchResponse response;
    response.add_results("result1 for " + request.query());
    response.add_results("this is result2");
    reply.send(response);
  }

};
}  // namespace examples

int main() {
  rpcz::application application;
  rpcz::server server(application);
  examples::SearchServiceImpl search_service;
  server.register_service(&search_service);
  cout << "Serving requests on port 5555." << endl;
  server.bind("tcp://*:5555");
  application.run();
}
