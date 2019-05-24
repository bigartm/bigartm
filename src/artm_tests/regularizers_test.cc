// Copyright 2017, Additive Regularization of Topic Models.

#include <memory>
#include <vector>

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
      item->add_transaction_start_index(item->transaction_start_index_size());
      item->add_token_weight(1.0);
    }
    item->add_transaction_start_index(item->transaction_start_index_size());
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

// artm_tests.exe --gtest_filter=Regularizers.NetPlsa
TEST(Regularizers, NetPlsa) {
  int nTopics = 8;
  int nTokens = 10;
  int nDocs = 5;

  // create master
  ::artm::MasterModelConfig master_config = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  master_config.set_cache_theta(true);

  // create regularizers
  ::artm::RegularizerConfig* regularizer_config = master_config.add_regularizer_config();

  regularizer_config->set_name("NetPlsaRegularizer_1");
  regularizer_config->set_type(::artm::RegularizerType_NetPlsaPhi);
  regularizer_config->set_tau(2.0f);

  ::artm::NetPlsaPhiConfig internal_config;
  internal_config.set_class_id("@time_class");
  internal_config.add_vertex_name("time_1");
  internal_config.add_vertex_name("time_2");
  internal_config.add_vertex_weight(2.0);
  internal_config.add_vertex_weight(1.0);
  internal_config.add_first_vertex_index(0);
  internal_config.add_second_vertex_index(1);
  internal_config.add_edge_weight(3.0);
  internal_config.set_symmetric_edge_weights(true);

  regularizer_config->set_config(internal_config.SerializeAsString());


  regularizer_config = master_config.add_regularizer_config();

  regularizer_config->set_name("NetPlsaRegularizer_2");
  regularizer_config->set_type(::artm::RegularizerType_NetPlsaPhi);
  regularizer_config->set_tau(-2.0f);

  ::artm::NetPlsaPhiConfig internal_config_2;
  internal_config_2.set_class_id("@time_class");
  internal_config_2.add_vertex_name("time_1");
  internal_config_2.add_vertex_name("time_2");
  internal_config_2.add_first_vertex_index(0);
  internal_config_2.add_second_vertex_index(1);
  internal_config_2.add_edge_weight(-3.0);
  internal_config_2.add_first_vertex_index(1);
  internal_config_2.add_second_vertex_index(0);
  internal_config_2.add_edge_weight(8.0);

  internal_config_2.set_symmetric_edge_weights(false);

  regularizer_config->set_config(internal_config_2.SerializeAsString());

  artm::MasterModel master(master_config);
  ::artm::test::Api api(master);

  // generate data
  artm::Batch batch;
  batch.set_id("11972762-6a23-4524-b089-7122816aff72");
  for (int i = 0; i < nTokens; i++) {
    std::stringstream str;
    str << "token" << i;
    batch.add_token(str.str());
    batch.add_class_id("@default_class");
  }
  batch.add_token("time_1");
  batch.add_class_id("@time_class");
  batch.add_token("time_2");
  batch.add_class_id("@time_class");

  for (int iDoc = 0; iDoc < nDocs; iDoc++) {
    artm::Item* item = batch.add_item();
    item->set_id(iDoc);
    for (int iToken = 0; iToken < nTokens; ++iToken) {
      item->add_token_id(iToken);
      item->add_transaction_start_index(item->transaction_start_index_size());
      int background_count = (iToken > 40) ? (1 + rand() % 5) : 0;  // NOLINT
      int topical_count = ((iToken < 40) && ((iToken % 10) == (iDoc % 10))) ? 10 : 0;
      item->add_token_weight(static_cast<float>(background_count + topical_count));
    }

    if (iDoc < 2) {
      item->add_token_id(nTokens);
      item->add_transaction_start_index(item->transaction_start_index_size());
      item->add_token_weight(1.0f);
    } else if (iDoc == 2) {
      item->add_token_id(nTokens + 1);
      item->add_transaction_start_index(item->transaction_start_index_size());
      item->add_token_weight(1.0f);
    }

    item->add_transaction_start_index(item->transaction_start_index_size());
  }

  // iterations
  auto offline_args = api.Initialize({ std::make_shared<artm::Batch>(batch) });
  for (int iter = 0; iter < 2; ++iter)
    master.FitOfflineModel(offline_args);

  // get and check theta
  ::artm::ThetaMatrix theta_matrix = master.GetThetaMatrix();

  std::vector<float> real_values;
  for (int j = 0; j < nDocs; ++j) {
    real_values.push_back(theta_matrix.item_weights(j).value(2));
  }

  std::vector<float> expected_values = { 0.000f, 0.000f, 0.000f, 0.000f, 0.999f };

  for (int i = 0; i < nDocs; ++i) {
    ASSERT_NEAR(real_values[i], expected_values[i], 1.0e-3);
  }
}

// artm_tests.exe --gtest_filter=Regularizers.RelativeRegularization
TEST(Regularizers, RelativeRegularization) {
  int nTopics = 50;
  int nTokens = 50;
  int nDocs = 100;

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
      item->add_transaction_start_index(item->transaction_start_index_size());
      item->add_token_weight(1.0);
    }
    item->add_transaction_start_index(item->transaction_start_index_size());
  }

  // part 1
  // create master
  ::artm::MasterModelConfig master_config = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  master_config.set_cache_theta(true);

  // create regularizer
  ::artm::RegularizerConfig* regularizer_config = master_config.add_regularizer_config();

  regularizer_config->set_name("SparsePhi");
  regularizer_config->set_type(::artm::RegularizerType_SmoothSparsePhi);
  regularizer_config->set_tau(-0.5);
  regularizer_config->set_gamma(0.5);

  regularizer_config->set_config(::artm::DecorrelatorPhiConfig().SerializeAsString());

  // create sparsity score
  ::artm::ScoreConfig* score_config = master_config.add_score_config();

  score_config->set_name("SparsityPhi");
  score_config->set_type(::artm::ScoreType_SparsityPhi);
  score_config->set_config(::artm::SparsityPhiScore().SerializeAsString());

  artm::MasterModel master(master_config);
  ::artm::test::Api api(master);

  std::vector<double> true_score = { 0.244, 0.380, 0.478, 0.544, 0.588,
                                     0.627, 0.665, 0.694, 0.716, 0.734,
                                     0.750, 0.768, 0.781, 0.790, 0.804,
                                     0.814, 0.824, 0.830, 0.836, 0.839 };

  auto offline_args = api.Initialize({ batch });
  for (int i = 0; i < 20; ++i) {
    master.FitOfflineModel(offline_args);

    ::artm::GetScoreArrayArgs args;
    args.set_score_name("SparsityPhi");

    auto sparsity_scores = master.GetScoreArrayAs< ::artm::SparsityPhiScore>(args);
    ASSERT_EQ(sparsity_scores.size(), (i + 1));
    ASSERT_NEAR(sparsity_scores.back().value(), true_score[i], 1e-3);
  }
}
