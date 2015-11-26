// Copyright 2014, Additive Regularization of Topic Models.

#include "boost/thread.hpp"
#include "gtest/gtest.h"

#include "boost/filesystem.hpp"

#include "artm/cpp_interface.h"
#include "artm/core/common.h"
#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/messages.pb.h"

#include "artm/core/internals.pb.h"
#include "artm/core/call_on_destruction.h"

#include "artm_tests/test_mother.h"

using artm::core::Helpers;
using artm::core::Token;

std::string runOfflineTest() {
  const int nTopics = 5;

  ::artm::MasterComponentConfig master_config;
  master_config.set_cache_theta(true);
  master_config.set_processors_count(1);
  ::artm::MasterComponent master_component(master_config);

  // Create model
  artm::ModelConfig model_config;
  model_config.set_topics_count(nTopics);
  model_config.set_name(artm::test::Helpers::getUniqueString());
  artm::Model model(master_component, model_config);

  std::vector<std::shared_ptr< ::artm::Batch>> batches;
  int batches_size = 2;
  int nTokens = 10;
  ::artm::test::TestMother::GenerateBatches(batches_size, nTokens, &batches);
  for (int iter = 0; iter < 3; ++iter) {
    for (unsigned iBatch = 0; iBatch < batches.size(); ++iBatch) {
      master_component.AddBatch(*batches[iBatch]);
    }

    master_component.WaitIdle();
    model.Synchronize(0.0);
  }

  std::shared_ptr< ::artm::TopicModel> topic_model = master_component.GetTopicModel(model.name());
  std::stringstream ss;
  ss << "Topic model:\n" << ::artm::test::Helpers::DescribeTopicModel(*topic_model);
  ss << "Theta matrix:\n";
  for (unsigned i = 0; i < batches.size(); ++i) {
    ::artm::GetThetaMatrixArgs args;
    args.set_model_name(model.name());
    args.mutable_batch()->CopyFrom(*batches[i]);
    std::shared_ptr< ::artm::ThetaMatrix> theta_matrix = master_component.GetThetaMatrix(args);
    ss << ::artm::test::Helpers::DescribeThetaMatrix(*theta_matrix);
  }

  return ss.str();
}

// artm_tests.exe --gtest_filter=RepeatableResult.Offline
TEST(RepeatableResult, Offline) {
  std::string first_result = runOfflineTest();
  std::string second_result = runOfflineTest();
  if (first_result != second_result)
    std::cout << first_result << "\n" << second_result;
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
  auto token_hasher = artm::core::TokenHasher();
  ASSERT_APPROX_EQ(Helpers::GenerateRandomVector(3, Token("class_id_1", ""))[0], 0.245338);
  ASSERT_APPROX_EQ(Helpers::GenerateRandomVector(3, Token("1_class_id", ""))[0], 0.319662);
  ASSERT_APPROX_EQ(Helpers::GenerateRandomVector(3, Token("", "token_1"))[0], 0.341962);
  ASSERT_APPROX_EQ(Helpers::GenerateRandomVector(3, Token("", "1_token"))[0], 0.315842);
  ASSERT_APPROX_EQ(Helpers::GenerateRandomVector(3, Token("class_id_1", "token_1"))[0], 0.318573);
  ASSERT_APPROX_EQ(Helpers::GenerateRandomVector(3, Token("class_id_2", "token_2"))[0], 0.410061);
}

void OverwriteTopicModel_internal(::artm::GetTopicModelArgs_RequestType request_type,
                                  ::artm::GetTopicModelArgs_MatrixLayout matrix_layout) {
  const int nTopics = 16;

  ::artm::MasterComponentConfig master_config;
  master_config.set_cache_theta(true);
  master_config.set_processors_count(1);
  ::artm::MasterComponent master_component(master_config);

  ::artm::RegularizerConfig sparse_phi_config;
  sparse_phi_config.set_name("sparse_phi");
  sparse_phi_config.set_type(::artm::RegularizerConfig_Type_SmoothSparsePhi);
  sparse_phi_config.set_config(::artm::SmoothSparsePhiConfig().SerializeAsString());
  ::artm::Regularizer sparse_phi(master_component, sparse_phi_config);

  // Create model
  artm::ModelConfig model_config;
  model_config.add_regularizer_name(sparse_phi_config.name());
  model_config.add_regularizer_tau(-0.05);
  for (int i = 0; i < nTopics; ++i) {
    std::stringstream ss;
    ss << "@topic_" << i;
    model_config.add_topic_name(ss.str());
  }

  model_config.set_name(artm::test::Helpers::getUniqueString());
  artm::Model model(master_component, model_config);

  std::vector<std::shared_ptr< ::artm::Batch>> batches;

  int batches_size = 2;
  int nTokens = 10;
  ::artm::test::TestMother::GenerateBatches(batches_size, nTokens, &batches);

  for (int iter = 0; iter < 3; ++iter) {
    for (unsigned iBatch = 0; iBatch < batches.size(); ++iBatch) {
      master_component.AddBatch(*batches[iBatch]);
    }

    master_component.WaitIdle();
    model.Synchronize(0.0);
  }

  ::artm::MasterComponent master2(master_config);
  ::artm::MasterComponent master3(master_config);
  ::artm::Regularizer sparse_phi2(master2, sparse_phi_config);
  ::artm::Regularizer sparse_phi3(master3, sparse_phi_config);
  ::artm::Model model2(master2, model_config);
  ::artm::Model model3(master3, model_config);

  ::artm::GetTopicModelArgs_RequestType request_types[2] = {
    artm::GetTopicModelArgs_RequestType_Pwt,
    artm::GetTopicModelArgs_RequestType_Nwt };

  bool pwt_request = request_type == artm::GetTopicModelArgs_RequestType_Pwt;
  bool nwt_request = !pwt_request;

  int slices = 3;
  for (int i = 0; i < slices; ++i) {
    ::artm::GetTopicModelArgs get_topic_model_args;
    get_topic_model_args.set_model_name(model.name());
    get_topic_model_args.set_request_type(request_type);
    get_topic_model_args.set_matrix_layout(matrix_layout);
    for (int topic_index = i; topic_index < nTopics; topic_index += slices) {
      get_topic_model_args.add_topic_name(model_config.topic_name(topic_index));
    }

    // To precisely overwrite topic model via n_wt counters one has manually
    // commit the model via model.Synchronize() method. This ensures that regularizers
    // are calculated for the model. Using "commit" argument gives wrong result
    // because it synchronizes model without regularizers. This is a reasonable
    // option when Overwrite is called through Pwt values, but for Nwt it is simply wrong.
    model2.Overwrite(*master_component.GetTopicModel(get_topic_model_args), false);
  }
  master2.WaitIdle();
  model2.Synchronize(/* decay_weight =*/ 0.0,
                      /* apply_weight =*/ 1.0,
                      /* invoke_regularizers =*/ nwt_request);  // invoke regularizers only for nwt_request
  std::string file_name = ::artm::test::Helpers::getUniqueString();
  artm::core::call_on_destruction c([&]() { try { boost::filesystem::remove(file_name); } catch (...) {} });
  model.Export(file_name);
  model3.Import(file_name);

  bool ok = false, ok2 = false;
  ::artm::test::Helpers::CompareTopicModels(*master2.GetTopicModel(model2.name()),
                                            *master_component.GetTopicModel(model.name()), &ok);
  ::artm::test::Helpers::CompareTopicModels(*master3.GetTopicModel(model3.name()),
                                            *master_component.GetTopicModel(model.name()), &ok2);
  if (!ok) {
    std::cout << "New topic model:\n"
              << ::artm::test::Helpers::DescribeTopicModel(*master2.GetTopicModel(model2.name()));
    std::cout << "Old topic model:\n"
              << ::artm::test::Helpers::DescribeTopicModel(*master_component.GetTopicModel(model.name()));
  }
  if (!ok2) {
    std::cout << "Imported topic model:\n"
              << ::artm::test::Helpers::DescribeTopicModel(*master3.GetTopicModel(model3.name()));
    std::cout << "Exported topic model:\n"
              << ::artm::test::Helpers::DescribeTopicModel(*master_component.GetTopicModel(model.name()));
  }
  ASSERT_TRUE(ok && ok2);
  for (unsigned iBatch = 0; iBatch < batches.size(); ++iBatch) {
    ::artm::test::Helpers::CompareThetaMatrices(*master2.GetThetaMatrix(model.name(), *batches[iBatch]),
                                                *master_component.GetThetaMatrix(model.name(), *batches[iBatch]), &ok);
    ::artm::test::Helpers::CompareThetaMatrices(*master3.GetThetaMatrix(model.name(), *batches[iBatch]),
                                                *master_component.GetThetaMatrix(model.name(), *batches[iBatch]), &ok2);
  }
  ASSERT_TRUE(ok && ok2);

  if (pwt_request)
    return;  // do not validate further model inference for pwt_request

  // Run extra iteration and validate that model is stil still the same
  for (unsigned iBatch = 0; iBatch < batches.size(); ++iBatch) {
    master_component.AddBatch(*batches[iBatch]);
    master2.AddBatch(*batches[iBatch]);
    master3.AddBatch(*batches[iBatch]);
  }

  master_component.WaitIdle(); master2.WaitIdle(); master3.WaitIdle();
  model.Synchronize(0.5); model2.Synchronize(0.5); model3.Synchronize(0.5);

  ::artm::test::Helpers::CompareTopicModels(*master2.GetTopicModel(model2.name()),
                                            *master_component.GetTopicModel(model.name()), &ok);
  ASSERT_TRUE(ok);

  ::artm::test::Helpers::CompareTopicModels(*master3.GetTopicModel(model3.name()),
                                            *master_component.GetTopicModel(model.name()), &ok2);
  ASSERT_TRUE(ok2);

  for (unsigned iBatch = 0; iBatch < batches.size(); ++iBatch) {
    ::artm::test::Helpers::CompareThetaMatrices(*master2.GetThetaMatrix(model.name(), *batches[iBatch]),
                                                *master_component.GetThetaMatrix(model.name(), *batches[iBatch]), &ok);
    ASSERT_TRUE(ok);

    ::artm::test::Helpers::CompareThetaMatrices(*master3.GetThetaMatrix(model.name(), *batches[iBatch]),
                                                *master_component.GetThetaMatrix(model.name(),
                                                *batches[iBatch]), &ok2);
    ASSERT_TRUE(ok2);
  }
}

// artm_tests.exe --gtest_filter=RepeatableResult.OverwriteTopicModel_Pwt_dense
TEST(RepeatableResult, OverwriteTopicModel_Pwt_dense) {
  OverwriteTopicModel_internal(artm::GetTopicModelArgs_RequestType_Pwt, artm::GetTopicModelArgs_MatrixLayout_Dense);
}

// artm_tests.exe --gtest_filter=RepeatableResult.OverwriteTopicModel_Pwt_sparse
TEST(RepeatableResult, OverwriteTopicModel_Pwt_sparse) {
  OverwriteTopicModel_internal(artm::GetTopicModelArgs_RequestType_Pwt, artm::GetTopicModelArgs_MatrixLayout_Sparse);
}

