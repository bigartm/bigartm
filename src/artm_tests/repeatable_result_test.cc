// Copyright 2017, Additive Regularization of Topic Models.

#include "boost/thread.hpp"
#include "gtest/gtest.h"

#include "boost/filesystem.hpp"

#include "artm/cpp_interface.h"
#include "artm/core/common.h"
#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/token.h"

#include "artm/core/call_on_destruction.h"

#include "artm_tests/test_mother.h"
#include "artm_tests/api.h"

using artm::core::Helpers;
using artm::core::Token;

std::string runOfflineTest() {
  const int nTopics = 5;

  ::artm::MasterModelConfig master_config = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  master_config.set_cache_theta(true);
  master_config.set_num_processors(1);
  master_config.set_pwt_name(artm::test::Helpers::getUniqueString());
  ::artm::MasterModel master_component(master_config);
  ::artm::test::Api api(master_component);

  int batches_size = 2;
  int nTokens = 10;
  auto batches = ::artm::test::TestMother::GenerateBatches(batches_size, nTokens);
  auto offline_args = api.Initialize(batches);

  for (int iter = 0; iter < 3; ++iter) {
    master_component.FitOfflineModel(offline_args);
  }

  ::artm::TopicModel topic_model = master_component.GetTopicModel();
  std::stringstream ss;
  ss << "Topic model:\n" << ::artm::test::Helpers::DescribeTopicModel(topic_model);
  ss << "Theta matrix:\n";
  for (unsigned i = 0; i < batches.size(); ++i) {
    ::artm::TransformMasterModelArgs args;
    args.add_batch_filename(batches[i]->id());
    args.set_theta_matrix_type(::artm::ThetaMatrixType_Dense);
    auto theta_matrix = master_component.Transform(args);
    ss << ::artm::test::Helpers::DescribeThetaMatrix(theta_matrix);
  }

  return ss.str();
}

// artm_tests.exe --gtest_filter=RepeatableResult.Offline
TEST(RepeatableResult, Offline) {
  std::string first_result = runOfflineTest();
  std::string second_result = runOfflineTest();
  if (first_result != second_result) {
    std::cout << first_result << "\n" << second_result;
  }
  ASSERT_EQ(first_result, second_result);
}

// artm_tests.exe --gtest_filter=RepeatableResult.RandomGenerator
TEST(RepeatableResult, RandomGenerator) {
  int num = 10;
  size_t seed = 5;
  std::vector<float> first_result = ::artm::core::Helpers::GenerateRandomVector(num, seed);
  std::vector<float> second_result = ::artm::core::Helpers::GenerateRandomVector(num, seed);
  ASSERT_EQ(first_result.size(), num);
  ASSERT_EQ(second_result.size(), num);
  for (int i = 1; i < num; ++i) {
    ASSERT_EQ(first_result[i], second_result[i]);
    ASSERT_NE(first_result[i - 1], first_result[i]);
  }
}

// artm_tests.exe --gtest_filter=RepeatableResult.TokenHasher
TEST(RepeatableResult, TokenHasher) {
  ASSERT_APPROX_EQ(Helpers::GenerateRandomVector(3, Token("class_id_1", ""))[0], 0.245338);
  ASSERT_APPROX_EQ(Helpers::GenerateRandomVector(3, Token("1_class_id", ""))[0], 0.319662);
  ASSERT_APPROX_EQ(Helpers::GenerateRandomVector(3, Token("", "token_1"))[0], 0.341962);
  ASSERT_APPROX_EQ(Helpers::GenerateRandomVector(3, Token("", "1_token"))[0], 0.315842);
  ASSERT_APPROX_EQ(Helpers::GenerateRandomVector(3, Token("class_id_1", "token_1"))[0], 0.318573);
  ASSERT_APPROX_EQ(Helpers::GenerateRandomVector(3, Token("class_id_2", "token_2"))[0], 0.410061);
}

void OverwriteTopicModel_internal(::artm::MatrixLayout matrix_layout) {
  const int nTopics = 16;
  int batches_size = 2;
  int nTokens = 10;

  auto master_config = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  master_config.set_cache_theta(true);
  master_config.set_num_processors(1);

  ::artm::RegularizerConfig* sparse_phi_config = master_config.add_regularizer_config();
  sparse_phi_config->set_name("sparse_phi");
  sparse_phi_config->set_type(::artm::RegularizerType_SmoothSparsePhi);
  sparse_phi_config->set_config(::artm::SmoothSparsePhiConfig().SerializeAsString());
  sparse_phi_config->set_tau(-0.05);

  ::artm::MasterModel master_component(master_config);
  ::artm::test::Api api(master_component);

  // Create model
  auto batches = ::artm::test::TestMother::GenerateBatches(batches_size, nTokens);
  ::artm::ImportBatchesArgs import_args;
  auto offline_args = api.Initialize(batches, &import_args);

  for (int iter = 0; iter < 3; ++iter) {
    master_component.FitOfflineModel(offline_args);
  }

  ::artm::MasterModel master2(master_config);
  master2.ImportBatches(import_args);
  ::artm::test::Api api2(master2);

  ::artm::MasterModel master3(master_config);
  master3.ImportBatches(import_args);
  ::artm::test::Api api3(master3);

  ::artm::GetTopicModelArgs get_topic_model_args;
  get_topic_model_args.set_model_name(master_config.pwt_name());
  get_topic_model_args.set_matrix_layout(matrix_layout);
  api2.OverwriteModel(master_component.GetTopicModel(get_topic_model_args));

  std::string file_name = ::artm::test::Helpers::getUniqueString();
  artm::core::call_on_destruction c([&]() { try { boost::filesystem::remove(file_name); } catch (...) { } });  // NOLINT
  ::artm::ExportModelArgs export_args;
  export_args.set_model_name(master_config.pwt_name());
  export_args.set_file_name(file_name);
  master_component.ExportModel(export_args);

  ::artm::ImportModelArgs import_model_args;
  import_model_args.set_model_name(master_config.pwt_name());
  import_model_args.set_file_name(file_name);
  master3.ImportModel(import_model_args);

  bool ok = false, ok2 = false;
  ::artm::test::Helpers::CompareTopicModels(master2.GetTopicModel(),
                                            master_component.GetTopicModel(), &ok);
  ::artm::test::Helpers::CompareTopicModels(master3.GetTopicModel(),
                                            master_component.GetTopicModel(), &ok2);
  if (!ok) {
    std::cout << "New topic model:\n"
              << ::artm::test::Helpers::DescribeTopicModel(master2.GetTopicModel());
    std::cout << "Old topic model:\n"
              << ::artm::test::Helpers::DescribeTopicModel(master_component.GetTopicModel());
  }
  if (!ok2) {
    std::cout << "Imported topic model:\n"
              << ::artm::test::Helpers::DescribeTopicModel(master3.GetTopicModel());
    std::cout << "Exported topic model:\n"
              << ::artm::test::Helpers::DescribeTopicModel(master_component.GetTopicModel());
  }
  ASSERT_TRUE(ok && ok2);
  for (unsigned iBatch = 0; iBatch < batches.size(); ++iBatch) {
    ::artm::TransformMasterModelArgs transform_args;
    transform_args.set_theta_matrix_type(::artm::ThetaMatrixType_Dense);
    transform_args.add_batch_filename(batches[iBatch]->id());
    ::artm::test::Helpers::CompareThetaMatrices(master2.Transform(transform_args),
                                                master_component.Transform(transform_args), &ok);
    ::artm::test::Helpers::CompareThetaMatrices(master3.Transform(transform_args),
                                                master_component.Transform(transform_args), &ok2);
  }
  ASSERT_TRUE(ok && ok2);

  // Run extra iteration and validate that model is stil still the same
  master_component.FitOfflineModel(offline_args);
  master2.FitOfflineModel(offline_args);
  master3.FitOfflineModel(offline_args);

  ::artm::test::Helpers::CompareTopicModels(master2.GetTopicModel(),
                                            master_component.GetTopicModel(), &ok);
  ASSERT_TRUE(ok);

  ::artm::test::Helpers::CompareTopicModels(master3.GetTopicModel(),
                                            master_component.GetTopicModel(), &ok2);
  ASSERT_TRUE(ok2);

  for (unsigned iBatch = 0; iBatch < batches.size(); ++iBatch) {
    ::artm::TransformMasterModelArgs transform_args;
    transform_args.set_theta_matrix_type(::artm::ThetaMatrixType_Dense);
    transform_args.add_batch_filename(batches[iBatch]->id());
    ::artm::test::Helpers::CompareThetaMatrices(master2.Transform(transform_args),
                                                master_component.Transform(transform_args), &ok);
    ::artm::test::Helpers::CompareThetaMatrices(master3.Transform(transform_args),
                                                master_component.Transform(transform_args), &ok2);
  }
  ASSERT_TRUE(ok && ok2);
}

// artm_tests.exe --gtest_filter=RepeatableResult.OverwriteTopicModel_Pwt_dense
TEST(RepeatableResult, OverwriteTopicModel_Pwt_dense) {
  OverwriteTopicModel_internal(artm::MatrixLayout_Dense);
}

// artm_tests.exe --gtest_filter=RepeatableResult.OverwriteTopicModel_Pwt_sparse
TEST(RepeatableResult, OverwriteTopicModel_Pwt_sparse) {
  OverwriteTopicModel_internal(artm::MatrixLayout_Sparse);
}

