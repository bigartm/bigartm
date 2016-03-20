// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_TESTS_TEST_MOTHER_H_
#define SRC_ARTM_TESTS_TEST_MOTHER_H_

#include <string>
#include <vector>

#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid.hpp"
#include "boost/uuid/uuid_generators.hpp"
#include "boost/uuid/uuid_io.hpp"

#include "artm/core/common.h"

#define ASSERT_APPROX_EQ(a, b) ASSERT_NEAR(a, b, (a + b) / 1e5)

namespace artm {
namespace test {

class Helpers {
 public:
  static std::string getUniqueString() {
    return boost::lexical_cast<std::string>(boost::uuids::random_generator()());
  }

  static std::string DescribeTopicModel(const ::artm::TopicModel& topic_model);
  static std::string DescribeThetaMatrix(const ::artm::ThetaMatrix& theta_matrix);
  static void CompareTopicModels(const ::artm::TopicModel& tm1, const ::artm::TopicModel& tm2, bool* ok);
  static void CompareThetaMatrices(const ::artm::ThetaMatrix& tm1, const ::artm::ThetaMatrix& tm2, bool *ok);
};

class TestMother {
 public:
  TestMother() : regularizer_name("regularizer1") {}
  RegularizerConfig GenerateRegularizerConfig() const;
  static MasterModelConfig GenerateMasterModelConfig(int nTopics);
  static std::vector<std::shared_ptr< ::artm::Batch>> GenerateBatches(
    int batches_size, int nTokens, ::artm::DictionaryData* dictionary = nullptr);
  static void GenerateBatches(int batches_size, int nTokens, const std::string& target_folder);

 private:
  const std::string regularizer_name;
};

}  // namespace test
}  // namespace artm

#endif  // SRC_ARTM_TESTS_TEST_MOTHER_H_
