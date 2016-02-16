// Copyright 2014, Additive Regularization of Topic Models.

#include <memory>

#include "gtest/gtest.h"

#include "artm/messages.pb.h"

#include "artm/cpp_interface.h"
#include "artm/core/instance.h"

#include "artm_tests/test_mother.h"
#include "artm_tests/api.h"

// artm_tests.exe --gtest_filter=Regularizers.TopicSelection
TEST(Regularizers, TopicSelection) {
  int nTopics = 10;

  // create master
  ::artm::MasterModelConfig master_config = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  master_config.set_cache_theta(true);

  // create regularizer
  ::artm::RegularizerConfig* regularizer_config = master_config.add_regularizer_config();

  regularizer_config->set_name("TopicSelectionRegularizer");
  regularizer_config->set_type(::artm::RegularizerConfig_Type_TopicSelectionTheta);
  regularizer_config->set_tau(0.5f);

  ::artm::TopicSelectionThetaConfig internal_config;
  for (int i = 0; i < nTopics; ++i)
    internal_config.add_topic_value(static_cast<float>(i) / nTopics);

  regularizer_config->set_config(internal_config.SerializeAsString());
  artm::MasterModel master(master_config);
  ::artm::test::Api api(master);

  // iterations
  auto batches = ::artm::test::TestMother::GenerateBatches(1, 5);
  auto offline_args = api.Initialize(batches);
  for (int iter = 0; iter < 3; ++iter)
    master.FitOfflineModel(offline_args);

  // get and check theta
  artm::GetThetaMatrixArgs args;
  ::artm::ThetaMatrix theta_matrix = master.GetThetaMatrix();

  // Uncomment to dump actual results
  // for (int i = 0; i <= 9; ++i)
  //  std::cout << theta_matrix.item_weights(0).value(i) << std::endl;
  float expected_values[] = { 0.41836, 0.262486, 0.160616, 0.0845677, 0.032849,
                              0.022987, 0.0103793, 0.0040327, 0.00267936, 0.00104289 };
  for (int i = 0; i < nTopics; ++i)
    ASSERT_NEAR(theta_matrix.item_weights(0).value(i), expected_values[i], 0.00001);
}
