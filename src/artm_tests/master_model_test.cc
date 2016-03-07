// Copyright 2014, Additive Regularization of Topic Models.

#include <vector>

#include "boost/filesystem.hpp"

#include "gtest/gtest.h"

#include "artm/cpp_interface.h"
#include "artm/messages.pb.h"

#include "artm_tests/test_mother.h"
#include "artm_tests/api.h"

void runBasicTest(bool skip_batch_dict) {
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
  ::artm::test::Api api(master_model);

  // Generate batches and load them into MasterModel
  ::artm::DictionaryData dictionary_data;
  const int nBatches = 20;
  const int nTokens = 30;
  auto batches = ::artm::test::TestMother::GenerateBatches(nBatches, nTokens, &dictionary_data);

  if (skip_batch_dict) {
    for (auto& batch : batches) {
      batch->clear_class_id();
      batch->clear_token();  // cast tokens away!!!
    }
  }

  ::artm::ImportBatchesArgs import_batches_args;
  for (auto& batch : batches) {
    import_batches_args.add_batch()->CopyFrom(*batch);
    import_batches_args.add_batch_name(batch->id());
  }
  master_model.ImportBatches(import_batches_args);

  if (skip_batch_dict) {
    try {
      ::artm::GatherDictionaryArgs gather_args;
      gather_args.set_dictionary_target_name("tmp_dict");
      gather_args.mutable_batch_path()->CopyFrom(import_batches_args.batch_name());
      master_model.GatherDictionary(gather_args);
      ASSERT_TRUE(false);  // exception expected because batches have no tokens
    }
    catch (const ::artm::InvalidOperationException& ex) {
    }
  }

  // Create dictionary
  dictionary_data.set_name("dictionary");
  master_model.CreateDictionary(dictionary_data);

  // Initialize model
  ::artm::InitializeModelArgs initialize_model_args;
  initialize_model_args.set_dictionary_name("dictionary");
  initialize_model_args.set_model_name(master_model.config().pwt_name());
  initialize_model_args.mutable_topic_name()->CopyFrom(master_model.config().topic_name());
  master_model.InitializeModel(initialize_model_args);

  ::artm::FitOfflineMasterModelArgs fit_offline_args;
  fit_offline_args.mutable_batch_filename()->CopyFrom(import_batches_args.batch_name());

  // Execute offline algorithm
  float expected[] = { 29.9952f, 26.1885f, 25.9853f, 24.5419f };
  for (int pass = 0; pass < 4; pass++) {
    master_model.FitOfflineModel(fit_offline_args);
    artm::PerplexityScore perplexity_score = master_model.GetScoreAs< ::artm::PerplexityScore>(get_score_args);
    ASSERT_APPROX_EQ(perplexity_score.value(), expected[pass]);
    // std::cout << "#" << pass << ": " << perplexity_score.value() << "\n";
  }

  const int update_every = 2;
  const float tau0 = 1024;
  const float kappa = 0.7;

  // Execute online algorithm
  float expected_sync[] = { 26.5443f, 26.3197f, 26.2796f, 26.2426f };
  float expected_async[] = { 27.2682f, 26.3178f, 26.2775f, 26.2407f };
  for (int is_async = 0; is_async <= 1; is_async++) {
    master_model.InitializeModel(initialize_model_args);
    float* expected = (is_async == 1) ? expected_async : expected_sync;
    int total_update_count = 0;
    for (int pass = 0; pass < 4; pass++) {
      ::artm::FitOnlineMasterModelArgs fit_online_args;
      fit_online_args.mutable_batch_filename()->CopyFrom(fit_offline_args.batch_filename());
      fit_online_args.set_async(is_async == 1);

      // Populate update_after and apply_weight fields
      int update_after = 0;
      do {
        total_update_count++;
        update_after += update_every;
        fit_online_args.add_update_after(std::min<int>(update_after, fit_online_args.batch_filename_size()));
        fit_online_args.add_apply_weight((total_update_count == 1) ? 1.0 : pow(tau0 + total_update_count, -kappa));
      } while (update_after < fit_online_args.batch_filename_size());

      master_model.FitOnlineModel(fit_online_args);
      artm::PerplexityScore perplexity_score = master_model.GetScoreAs< ::artm::PerplexityScore>(get_score_args);
      ASSERT_APPROX_EQ(perplexity_score.value(), expected[pass]);
    }

    ::artm::TransformMasterModelArgs transform_args;
    transform_args.mutable_batch()->CopyFrom(import_batches_args.batch());
    ::artm::ThetaMatrix theta = master_model.Transform(transform_args);
    ASSERT_EQ(theta.item_id_size(), nBatches);  // test_mother generates one item per batch
    ASSERT_EQ(theta.item_weights_size(), nBatches);
    ASSERT_EQ(theta.item_weights(0).value_size(), config.topic_name_size());
  }
}

// To run this particular test:
// artm_tests.exe --gtest_filter=MasterModel.Basic
TEST(MasterModel, Basic) {
  runBasicTest(/*skip_batch_dict=*/ false);
}

// To run this particular test:
// artm_tests.exe --gtest_filter=MasterModel.SkipBatchDict
TEST(MasterModel, SkipBatchDict) {
  runBasicTest(/*skip_batch_dict=*/ true);
}
