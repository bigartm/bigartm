// Copyright 2017, Additive Regularization of Topic Models.

#include <memory>

#include "gtest/gtest.h"

#include "artm/cpp_interface.h"
#include "artm/core/common.h"
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
  regularizer_config->set_type(::artm::RegularizerType_TopicSelectionTheta);
  regularizer_config->set_tau(0.5f);

  ::artm::TopicSelectionThetaConfig internal_config;
  for (int i = 0; i < nTopics; ++i) {
    internal_config.add_topic_value(static_cast<float>(i) / nTopics);
  }

  regularizer_config->set_config(internal_config.SerializeAsString());
  artm::MasterModel master(master_config);
  ::artm::test::Api api(master);

  // iterations
  auto batches = ::artm::test::TestMother::GenerateBatches(1, 5);
  auto offline_args = api.Initialize(batches);
  for (int iter = 0; iter < 3; ++iter) {
    master.FitOfflineModel(offline_args);
  }

  // get and check theta
  artm::GetThetaMatrixArgs args;
  ::artm::ThetaMatrix theta_matrix = master.GetThetaMatrix();

  // Uncomment to dump actual results
  // for (int i = 0; i <= 9; ++i)
  //  std::cout << theta_matrix.item_weights(0).value(i) << std::endl;
  float expected_values[] = { 0.41836f, 0.262486f, 0.160616f, 0.0845677f, 0.032849f,
                              0.022987f, 0.0103793f, 0.0040327f, 0.00267936f, 0.00104289f };
  for (int i = 0; i < nTopics; ++i) {
    ASSERT_NEAR(theta_matrix.item_weights(0).value(i), expected_values[i], 0.00001);
  }
}

// artm_tests.exe --gtest_filter=Regularizers.SmoothSparseTheta
TEST(Regularizers, SmoothSparseTheta) {
  int nTopics = 4;
  int nTokens = 5;
  int nDocs = 3;

  // generate batch
  std::shared_ptr<::artm::Batch> batch(new ::artm::Batch());
  batch->set_id(artm::test::Helpers::getUniqueString());

  for (int i = 0; i < nTokens; i++) {
    std::stringstream str;
    str << "token" << i;
    batch->add_token(str.str());
  }

  for (int i = 0; i < nDocs; ++i) {
    artm::Item* item = batch->add_item();
    std::stringstream str;
    str << "item_" << i;
    item->set_title(str.str());
    for (int iToken = 0; iToken < nTokens; ++iToken) {
      item->add_token_id(iToken);
      item->add_token_weight(1.0);
    }
  }

  // part 1
  // create master
  ::artm::MasterModelConfig master_config = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  master_config.set_cache_theta(true);

  // create regularizer
  ::artm::RegularizerConfig* regularizer_config = master_config.add_regularizer_config();

  regularizer_config->set_name("SSTRegularizer_1");
  regularizer_config->set_type(::artm::RegularizerType_SmoothSparseTheta);
  regularizer_config->set_tau(-100.0f);

  ::artm::SmoothSparseThetaConfig internal_config;
  internal_config.add_item_title("item_0");
  internal_config.add_item_title("item_2");

  regularizer_config->set_config(internal_config.SerializeAsString());

  artm::MasterModel master(master_config);
  ::artm::test::Api api(master);

  auto offline_args = api.Initialize({ batch });
  master.FitOfflineModel(offline_args);

  // get and check theta
  ::artm::ThetaMatrix theta_matrix = master.GetThetaMatrix();

  // nDocs x nTopics
  std::vector<std::vector<float> > expected_values = {
    { 0.0f,  0.0f,   0.0f,   0.0f },
    { 0.265f, 0.224f, 0.247f, 0.264f },
    { 0.0f,  0.0f,   0.0f,   0.0f }
  };

  for (int i = 0; i < nDocs; ++i) {
    for (int j = 0; j < nTopics; ++j) {
      ASSERT_NEAR(theta_matrix.item_weights(i).value(j), expected_values[i][j], 0.001);
    }
  }

  for (int i = 0; i < nDocs; ++i) {
    for (int j = 0; j < nTopics; ++j) {
      std::cout << theta_matrix.item_weights(i).value(j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // part 2
  // create master
  master_config = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  master_config.set_opt_for_avx(false);
  master_config.set_cache_theta(true);

  // create regularizer
  regularizer_config = master_config.add_regularizer_config();

  regularizer_config->set_name("SSTRegularizer_1");
  regularizer_config->set_type(::artm::RegularizerType_SmoothSparseTheta);
  regularizer_config->set_tau(100.0f);

  ::artm::SmoothSparseThetaConfig internal_config_2;
  internal_config_2.add_item_title("item_0");
  auto values = internal_config_2.add_item_topic_multiplier();
  values->add_value(1.0);
  values->add_value(0.0);
  values->add_value(1.0);
  values->add_value(0.0);

  internal_config_2.add_item_title("item_2");
  values = internal_config_2.add_item_topic_multiplier();
  for (int i = 0; i < nTopics; ++i) {
    values->add_value(-1.0f);
  }

  regularizer_config->set_config(internal_config_2.SerializeAsString());

  master.Reconfigure(master_config);
  ::artm::test::Api api_2(master);

  offline_args = api_2.Initialize({ batch });
  master.FitOfflineModel(offline_args);

  // get and check theta
  theta_matrix = master.GetThetaMatrix();

  // nDocs x nTopics
  expected_values = {
    { 0.5f,  0.0f,   0.5f,   0.0f },
    { 0.265f, 0.224f, 0.247f, 0.264f },
    { 0.0f,  0.0f,   0.0f,   0.0f }
  };

  for (int i = 0; i < nDocs; ++i) {
    for (int j = 0; j < nTopics; ++j) {
      ASSERT_NEAR(theta_matrix.item_weights(i).value(j), expected_values[i][j], 0.001);
    }
  }

  for (int i = 0; i < nDocs; ++i) {
    for (int j = 0; j < nTopics; ++j) {
      std::cout << theta_matrix.item_weights(i).value(j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
