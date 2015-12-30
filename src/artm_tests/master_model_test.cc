// Copyright 2014, Additive Regularization of Topic Models.

#include <vector>

#include "gtest/gtest.h"

#include "artm/cpp_interface.h"
#include "artm/messages.pb.h"

#include "artm_tests/test_mother.h"

// To run this particular test:
// artm_tests.exe --gtest_filter=MasterModel.Basic
TEST(MasterModel, Basic) {
  // Configure MasterModel
  ::artm::MasterModelConfig config;
  config.set_threads(2);
  config.set_pwt_name("pwt");
  config.add_topic_name("topic1"); config.add_topic_name("topic2");
  ::artm::ScoreConfig* score_config = config.add_score_config();
  score_config->set_type(::artm::ScoreConfig_Type_Perplexity);
  score_config->set_name("Perplexity");
  score_config->set_config(::artm::PerplexityScoreConfig().SerializeAsString());

  ::artm::GetScoreValueArgs get_score_args;
  get_score_args.set_model_name("pwt");
  get_score_args.set_score_name("Perplexity");

  ::artm::RegularizerConfig* reg_theta = config.add_regularizer_config();
  reg_theta->set_type(::artm::RegularizerConfig_Type_SmoothSparseTheta);
  reg_theta->set_tau(-0.2);
  reg_theta->set_name("SparseTheta");
  reg_theta->set_config(::artm::SmoothSparseThetaConfig().SerializeAsString());

  ::artm::RegularizerConfig* reg_phi = config.add_regularizer_config();
  reg_phi->set_type(::artm::RegularizerConfig_Type_SmoothSparsePhi);
  reg_phi->set_tau(-0.1);
  reg_phi->set_name("SparsePhi");
  reg_phi->set_config(::artm::SmoothSparsePhiConfig().SerializeAsString());

  // Create MasterModel
  ::artm::MasterModel master_model(config);

  // Generate batches and load them into MasterModel
  ::artm::ImportBatchesArgs import_batches_args;
  ::artm::DictionaryData dictionary_data;
  ::artm::test::TestMother::GenerateBatches(/*nBatches =*/ 20, /* nTokens =*/ 30,
                                            &import_batches_args, &dictionary_data);
  master_model.ImportBatches(import_batches_args);

  // Create dictionary
  dictionary_data.set_name("dictionary");
  master_model.CreateDictionary(dictionary_data);

  // Initialize model
  ::artm::InitializeModelArgs initialize_model_args;
  initialize_model_args.set_dictionary_name("dictionary");
  initialize_model_args.set_model_name("pwt");
  initialize_model_args.mutable_topic_name()->CopyFrom(config.topic_name());
  master_model.InitializeModel(initialize_model_args);

  // Execute offline algorithm
  ::artm::FitOfflineMasterModelArgs fit_offline_args;
  fit_offline_args.mutable_batch_filename()->CopyFrom(import_batches_args.batch_name());

  float expected[] = { 29.9952f, 26.1885f, 25.9853f, 24.5419f };
  for (int pass = 0; pass < 4; pass++) {
    master_model.FitOfflineModel(fit_offline_args);
    artm::PerplexityScore perplexity_score = master_model.GetScoreAs< ::artm::PerplexityScore>(get_score_args);
    ASSERT_APPROX_EQ(perplexity_score.value(), expected[pass]);
    // std::cout << "#" << pass << ": " << perplexity_score.value() << "\n";
  }

  ::artm::FitOnlineMasterModelArgs fit_online_args;
  fit_online_args.set_update_every(2);
  fit_online_args.mutable_batch_filename()->CopyFrom(import_batches_args.batch_name());

  // Execute online algorithm
  master_model.InitializeModel(initialize_model_args);
  float expected2[] = { 26.5443f, 26.3197f, 26.2796f, 26.2426f };
  fit_online_args.set_async(false);
  for (int pass = 0; pass < 4; pass++) {
    master_model.FitOnlineModel(fit_online_args);
    artm::PerplexityScore perplexity_score = master_model.GetScoreAs< ::artm::PerplexityScore>(get_score_args);
    ASSERT_APPROX_EQ(perplexity_score.value(), expected2[pass]);
  }

  // Execute async online algorithm
  master_model.InitializeModel(initialize_model_args);
  fit_online_args.set_async(true);
  float expected3[] = { 27.1744, 26.1734, 26.1399, 26.106 };
  for (int pass = 0; pass < 4; pass++) {
    master_model.FitOnlineModel(fit_online_args);

    artm::PerplexityScore perplexity_score = master_model.GetScoreAs< ::artm::PerplexityScore>(get_score_args);
    ASSERT_APPROX_EQ(perplexity_score.value(), expected3[pass]);
  }
}
