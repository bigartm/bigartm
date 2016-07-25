// Copyright 2015, Additive Regularization of Topic Models.
#include <fstream>

#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"

#include "gtest/gtest.h"

#include "artm/core/token.h"

// artm_tests.exe --gtest_filter=Serializer.Token
TEST(Serializer, Token) {
  // http://stackoverflow.com/questions/3015582/direct-boost-serialization-to-char-array
  std::ofstream ofs("filename");
  ::artm::core::Token token("my class", "my keyword");

  {
    boost::archive::text_oarchive oa(ofs);
    oa << token;
  }

  ::artm::core::Token token2;
  {
    std::ifstream ifs("filename");
    boost::archive::text_iarchive ia(ifs);
    ia >> token2;
  }

  ASSERT_EQ(token, token2);
}
