// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <string>
#include <vector>

#include "boost/filesystem/path.hpp"
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

  static artm::Batch GenerateBatch(int nTokens, int nDocs, const std::string& class1, const std::string& class2);
  static artm::DictionaryData GenerateDictionary(int nTokens, const std::string& class1, const std::string& class2);
  static void ConfigurePerplexityScore(const std::string& score_name,
                                       artm::MasterModelConfig* master_config,
                                       const std::vector<std::string>& class_ids = { },
                                       const std::vector<std::string>& tt_names = { });
  static boost::filesystem::path getTestDataDir();
};

class TestMother {
 public:
  TestMother() : regularizer_name("regularizer1") { }
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
