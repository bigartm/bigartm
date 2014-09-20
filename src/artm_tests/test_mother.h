// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_TESTS_TEST_MOTHER_H_
#define SRC_ARTM_TESTS_TEST_MOTHER_H_

#include <string>

#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid.hpp"
#include "boost/uuid/uuid_generators.hpp"
#include "boost/uuid/uuid_io.hpp"

#include "artm/core/common.h"
#include "artm/messages.pb.h"

namespace artm {
namespace test {

class TestMother {
 public:
  TestMother() : nTopics(10), regularizer_name("regularizer1") {}
  ModelConfig GenerateModelConfig() const;
  RegularizerConfig GenerateRegularizerConfig() const;

 private:
  const int nTopics;
  const std::string regularizer_name;
};

}  // namespace test
}  // namespace artm

#endif  // SRC_ARTM_TESTS_TEST_MOTHER_H_
