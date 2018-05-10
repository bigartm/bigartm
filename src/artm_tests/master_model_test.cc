// Copyright 2017, Additive Regularization of Topic Models.

#include <vector>

#include "boost/filesystem.hpp"

#include "gtest/gtest.h"

#include "artm/cpp_interface.h"
#include "artm/core/common.h"

#include "artm_tests/test_mother.h"
#include "artm_tests/api.h"

// To run this particular test:
// artm_tests.exe --gtest_filter=MasterModel.Basic
TEST(MasterModel, Basic) {
  // Configure MasterModel
  ::artm::MasterModelConfig config;
  config.set_num_processors(2);
  config.set_pwt_name("pwt");
  config.add_topic_name("topic1"); config.add_topic_name("topic2");
  ::artm::ScoreConfig* score_config = config.add_score_config();
  score_config->set_type(::artm::ScoreType_Perplexity);
  score_config->set_name("Perplexity");
  score_config->set_config(::artm::PerplexityScoreConfig().SerializeAsString());

  ::artm::ScoreConfig* score_config2 = config.add_score_config();
  score_config2->set_type(::artm::ScoreType_SparsityPhi);
  score_config2->set_name("SparsityPhi");
  score_config2->set_config(::artm::SparsityPhiScoreConfig().SerializeAsString());

  ::artm::GetScoreValueArgs get_score_args;
  get_score_args.set_score_name("Perplexity");

  ::artm::GetScoreArrayArgs get_score_array_args;
  get_score_array_args.set_score_name("Perplexity");

  ::artm::GetScoreArrayArgs get_score_array_args2;
  get_score_array_args2.set_score_name("SparsityPhi");

  ::artm::RegularizerConfig* reg_theta = config.add_regularizer_config();
  reg_theta->set_type(::artm::RegularizerType_SmoothSparseTheta);
  reg_theta->set_tau(-0.2);
  reg_theta->set_name("SparseTheta");
  reg_theta->set_config(::artm::SmoothSparseThetaConfig().SerializeAsString());

  ::artm::RegularizerConfig* reg_phi = config.add_regularizer_config();
  reg_phi->set_type(::artm::RegularizerType_SmoothSparsePhi);
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

  ::artm::ImportBatchesArgs import_batches_args;
  ::artm::GatherDictionaryArgs gather_args;
  for (auto& batch : batches) {
    import_batches_args.add_batch()->CopyFrom(*batch);
    gather_args.add_batch_path(batch->id());
  }
  master_model.ImportBatches(import_batches_args);

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
  fit_offline_args.mutable_batch_filename()->CopyFrom(gather_args.batch_path());

  // Execute offline algorithm
  float expected[] = { 29.9952f, 26.1885f, 25.9853f, 24.5419f };
  for (int pass = 0; pass < 4; pass++) {
    master_model.FitOfflineModel(fit_offline_args);
    artm::PerplexityScore perplexity_score = master_model.GetScoreAs< ::artm::PerplexityScore>(get_score_args);
    ASSERT_APPROX_EQ(perplexity_score.value(), expected[pass]);
    // std::cout << "#" << pass << ": " << perplexity_score.value() << "\n";

    auto perplexity_scores = master_model.GetScoreArrayAs< ::artm::PerplexityScore>(get_score_array_args);
    ASSERT_EQ(perplexity_scores.size(), (pass + 1));
    ASSERT_APPROX_EQ(perplexity_scores.back().value(), perplexity_score.value());

    auto sparsity_phi_scores = master_model.GetScoreArrayAs< ::artm::SparsityPhiScore>(get_score_array_args2);
    ASSERT_EQ(sparsity_phi_scores.size(), (pass + 1));
  }

  api.ClearScoreArrayCache(::artm::ClearScoreArrayCacheArgs());

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

      if (!fit_online_args.async()) {
        auto perplexity_scores = master_model.GetScoreArrayAs< ::artm::PerplexityScore>(get_score_array_args);
        ASSERT_EQ(perplexity_scores.size(), (pass + 1) * nBatches / update_every);

        auto sparsity_phi_scores = master_model.GetScoreArrayAs< ::artm::SparsityPhiScore>(get_score_array_args2);
        ASSERT_EQ(sparsity_phi_scores.size(), (pass + 1) * nBatches / update_every);
      }
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
// artm_tests.exe --gtest_filter=MasterModel.TestEmptyMasterModel
TEST(MasterModel, TestEmptyMasterModel) {
  ::artm::MasterModelConfig config;
  config.set_num_processors(0);
  ::artm::MasterModel model(config);
  auto info = model.info();
  EXPECT_EQ(info.num_processors(), 0);
}

// To run this particular test:
// artm_tests.exe --gtest_filter=MasterModel.TestClone
TEST(MasterModel, TestClone) {
  // Configure MasterModel
  ::artm::MasterModelConfig config;
  config.set_num_processors(2);
  config.set_cache_theta(true);
  config.add_topic_name("topic1"); config.add_topic_name("topic2");
  ::artm::ScoreConfig* score_config = config.add_score_config();
  score_config->set_type(::artm::ScoreType_Perplexity);
  score_config->set_name("Perplexity");
  score_config->set_config(::artm::PerplexityScoreConfig().SerializeAsString());

  ::artm::GetScoreValueArgs get_score_args;
  get_score_args.set_score_name("Perplexity");

  ::artm::GetScoreArrayArgs get_score_array_args;
  get_score_array_args.set_score_name("Perplexity");

  // Create MasterModel
  ::artm::MasterModel master_model(config);
  ::artm::test::Api api(master_model);

  // Generate batches and load them into MasterModel
  ::artm::DictionaryData dictionary_data;
  const int nBatches = 20;
  const int nTokens = 30;
  auto batches = ::artm::test::TestMother::GenerateBatches(nBatches, nTokens, &dictionary_data);

  ::artm::ImportBatchesArgs import_batches_args;
  ::artm::GatherDictionaryArgs gather_args;
  for (auto& batch : batches) {
    import_batches_args.add_batch()->CopyFrom(*batch);
    gather_args.add_batch_path(batch->id());
  }
  master_model.ImportBatches(import_batches_args);

  // Create dictionary
  dictionary_data.set_name("dictionary");
  master_model.CreateDictionary(dictionary_data);

  // Initialize model
  ::artm::InitializeModelArgs initialize_model_args;
  initialize_model_args.set_dictionary_name("dictionary");
  initialize_model_args.set_model_name(master_model.config().pwt_name());
  initialize_model_args.mutable_topic_name()->CopyFrom(master_model.config().topic_name());
  master_model.InitializeModel(initialize_model_args);

  // Execute offline algorithm
  ::artm::FitOfflineMasterModelArgs fit_offline_args;
  fit_offline_args.mutable_batch_filename()->CopyFrom(gather_args.batch_path());
  fit_offline_args.set_num_collection_passes(4);
  master_model.FitOfflineModel(fit_offline_args);

  int master_id = api.Duplicate(::artm::DuplicateMasterComponentArgs());
  artm::MasterModel master_clone(master_id);

  ASSERT_TRUE(master_clone.GetThetaMatrix().SerializeAsString() == master_model.GetThetaMatrix().SerializeAsString());
  ASSERT_TRUE(master_clone.GetTopicModel().SerializeAsString() == master_model.GetTopicModel().SerializeAsString());
  ASSERT_TRUE(
    master_clone.GetScore(get_score_args).SerializeAsString() ==
    master_model.GetScore(get_score_args).SerializeAsString());
  ASSERT_TRUE(
    master_clone.GetScoreArray(get_score_array_args).SerializeAsString() ==
    master_model.GetScoreArray(get_score_array_args).SerializeAsString());
}

void testReshapeTokens(bool with_ptdw, bool opt_for_avx) {
  ::artm::MasterModelConfig config;
  config.set_num_processors(2);
  config.set_pwt_name("pwt");
  config.add_topic_name("topic1"); config.add_topic_name("topic2");
  config.set_opt_for_avx(opt_for_avx);

  ::artm::ScoreConfig* score_config = config.add_score_config();
  score_config->set_type(::artm::ScoreType_Perplexity);
  score_config->set_name("Perplexity");
  score_config->set_config(::artm::PerplexityScoreConfig().SerializeAsString());

  ::artm::RegularizerConfig* reg_phi = config.add_regularizer_config();
  reg_phi->set_type(::artm::RegularizerType_SmoothSparsePhi);
  reg_phi->set_tau(0.1);
  reg_phi->set_name("SmoothPhi");
  reg_phi->set_config(::artm::SmoothSparsePhiConfig().SerializeAsString());

  if (with_ptdw) {
    // Create ptdw-regularizer
    ::artm::RegularizerConfig* regularizer_config2 = config.add_regularizer_config();
    regularizer_config2->set_name("regularizer_ptdw");
    regularizer_config2->set_type(::artm::RegularizerType_SmoothPtdw);
    ::artm::SmoothPtdwConfig smooth_ptdw_config;
    smooth_ptdw_config.set_window(5);
    regularizer_config2->set_config(smooth_ptdw_config.SerializeAsString());
    regularizer_config2->set_tau(0.0);
  }

  ::artm::GetScoreValueArgs get_score_args;
  get_score_args.set_score_name("Perplexity");

  // Create MasterModel
  ::artm::MasterModel master_model(config);
  ::artm::test::Api api(master_model);

  // Generate batches and load them into MasterModel
  ::artm::DictionaryData full_dict;
  const int nBatches = 20;
  const int nTokens = 30;
  auto batches = ::artm::test::TestMother::GenerateBatches(nBatches, nTokens, &full_dict);

  // Pick each second word from the dictionary
  ::artm::DictionaryData small_dict;
  for (int iToken = 0; iToken < nTokens; iToken += 2) {
    small_dict.add_token(full_dict.token(iToken));
  }
  auto fit_offline_args = api.Initialize(batches, nullptr, nullptr, &small_dict);

  float expected[] = { 14.3481f, 11.7418f, 10.8133f, 10.3792f };
  for (int pass = 0; pass < 4; pass++) {
    master_model.FitOfflineModel(fit_offline_args);
    artm::PerplexityScore perplexity_score = master_model.GetScoreAs< ::artm::PerplexityScore>(get_score_args);
    // ASSERT_APPROX_EQ(perplexity_score.value(), expected[pass]);
  }

  ::artm::GetTopicModelArgs get_pwt_args;
  get_pwt_args.set_model_name(master_model.config().pwt_name());
  ::artm::GetTopicModelArgs get_nwt_args;
  get_nwt_args.set_model_name(master_model.config().nwt_name());
  ::artm::TopicModel nwt_model = master_model.GetTopicModel(get_nwt_args);
  ASSERT_EQ(nwt_model.token_size(), small_dict.token_size());

  // Reshape nwt model
  full_dict.set_name("full_dict");
  master_model.CreateDictionary(full_dict);
  ::artm::InitializeModelArgs init_model_args;
  init_model_args.set_dictionary_name(full_dict.name());
  init_model_args.set_model_name(master_model.config().nwt_name());
  master_model.InitializeModel(init_model_args);
  nwt_model = master_model.GetTopicModel(get_nwt_args);
  ASSERT_EQ(nwt_model.token_size(), full_dict.token_size());
  ::artm::TopicModel pwt_model = master_model.GetTopicModel(get_pwt_args);
  ASSERT_EQ(pwt_model.token_size(), small_dict.token_size());

  master_model.FitOfflineModel(fit_offline_args);
  nwt_model = master_model.GetTopicModel(get_nwt_args);
  ASSERT_EQ(nwt_model.token_size(), full_dict.token_size());
  pwt_model = master_model.GetTopicModel(get_pwt_args);
  ASSERT_EQ(pwt_model.token_size(), full_dict.token_size());

  // Validate that new tokens were initialized
  for (int token_index = 0; token_index < nTokens; token_index++) {
    for (int topic_index = 0; topic_index < 2; topic_index++) {
      ASSERT_GT(nwt_model.token_weights(token_index).value(topic_index), 0);
      ASSERT_GT(pwt_model.token_weights(token_index).value(topic_index), 0);
    }
  }

  // Hard-code expected values
  // std::cout << ::artm::test::Helpers::DescribeTopicModel(pwt_model);
  // std::cout << ::artm::test::Helpers::DescribeTopicModel(nwt_model);
  ASSERT_APPROX_EQ(nwt_model.token_weights(nTokens - 1).value(0), 1.39982);
  ASSERT_APPROX_EQ(nwt_model.token_weights(nTokens - 1).value(1), 5.60018);
  ASSERT_APPROX_EQ(pwt_model.token_weights(nTokens - 1).value(0), 0.0075085);
  ASSERT_APPROX_EQ(pwt_model.token_weights(nTokens - 1).value(1), 0.0330261);
}

// To run this particular test:
// artm_tests.exe --gtest_filter=MasterModel.TestReshapeTokensAvxOn
TEST(MasterModel, TestReshapeTokensAvxOn) {
  testReshapeTokens(false, true);
}

// To run this particular test:
// artm_tests.exe --gtest_filter=MasterModel.TestReshapeTokensAvxOff
TEST(MasterModel, TestReshapeTokensAvxOff) {
  testReshapeTokens(false, false);
}

// To run this particular test:
// artm_tests.exe --gtest_filter=MasterModel.TestReshapeTokensPtdw
TEST(MasterModel, TestReshapeTokensPtdw) {
  testReshapeTokens(true, true);
}

template<typename RegularizerConfig>
void testReorderTokens(::artm::RegularizerType regularizer_type, RegularizerConfig reg_config, float tau) {
  bool opt_for_avx = true;

  int nTokens = 60;
  int nDocs = 100;
  int nTopics = 10;

  ::artm::MasterModelConfig config = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  config.set_num_processors(2);
  config.set_pwt_name("pwt");
  config.add_class_id("@default_class"); config.add_class_weight(0.5f);
  config.add_class_id("__custom_class"); config.add_class_weight(2.0f);
  config.set_opt_for_avx(opt_for_avx);

  ::artm::ScoreConfig* score_config = config.add_score_config();
  score_config->set_type(::artm::ScoreType_Perplexity);
  score_config->set_name("Perplexity");
  score_config->set_config(::artm::PerplexityScoreConfig().SerializeAsString());

  ::artm::RegularizerConfig* reg_phi = config.add_regularizer_config();
  reg_phi->set_type(regularizer_type);
  reg_phi->set_tau(tau);
  reg_phi->set_name("MyRegularizer");
  reg_phi->set_config(reg_config.SerializeAsString());

  // Generate doc-token matrix
  artm::Batch batch = ::artm::test::Helpers::GenerateBatch(nTokens, nDocs, "@default_class", "__custom_class");
  std::vector<std::shared_ptr< ::artm::Batch>> batches;
  batches.push_back(std::make_shared< ::artm::Batch>(batch));

  ::artm::GetScoreValueArgs get_score_args;
  get_score_args.set_score_name("Perplexity");

  // Create MasterModel
  ::artm::MasterModel master_model(config);
  ::artm::test::Api api(master_model);

  // Create Master model where we will permute words in the dictionary
  ::artm::MasterModel master_model_perm(config);
  ::artm::test::Api api_perm(master_model_perm);

  // Generate batches and load them into MasterModel
  ::artm::DictionaryData full_dict;
  full_dict.mutable_token()->CopyFrom(batch.token());
  full_dict.mutable_class_id()->CopyFrom(batch.class_id());
  auto fit_offline_args = api.Initialize(batches, nullptr, nullptr, &full_dict);
  auto fit_offline_args_perm = api_perm.Initialize(batches, nullptr, nullptr, &full_dict);

  int shifts[] = { 0, 15, 6, 22 };
  for (int pass = 0; pass < 4; pass++) {
    if (pass > 0) {
      // Reorder words in the dictionary
      ::artm::DictionaryData perm_dict;
      for (int iToken = 0; iToken < nTokens; iToken++) {
        perm_dict.add_token(full_dict.token((shifts[pass] + iToken) % nTokens));
        perm_dict.add_class_id(full_dict.class_id((shifts[pass] + iToken) % nTokens));
      }
      // Permute words in the model
      perm_dict.set_name("perm_dict");
      master_model_perm.CreateDictionary(perm_dict);
      ::artm::InitializeModelArgs init_model_args;
      init_model_args.set_dictionary_name(perm_dict.name());
      init_model_args.set_model_name(master_model_perm.config().nwt_name());
      master_model_perm.InitializeModel(init_model_args);
    }
    master_model.FitOfflineModel(fit_offline_args);
    master_model_perm.FitOfflineModel(fit_offline_args_perm);
    auto perplexity_score = master_model.GetScoreAs< ::artm::PerplexityScore>(get_score_args);

    auto perplexity_score_perm = master_model_perm.GetScoreAs< ::artm::PerplexityScore>(get_score_args);
    ASSERT_APPROX_EQ(perplexity_score.value(), perplexity_score_perm.value());
  }

  // Make smaller dictionary to validate that regularizers handle pwt words that are not in nwt
  ::artm::DictionaryData small_dict;
  small_dict.set_name("small_dict");
  for (int iToken = 0; iToken < nTokens; iToken += 3) {
    small_dict.add_token(full_dict.token(iToken));
  }
  master_model.CreateDictionary(small_dict);
  ::artm::InitializeModelArgs init_model_args;
  init_model_args.set_dictionary_name(small_dict.name());
  init_model_args.set_model_name(master_model.config().nwt_name());
  master_model.InitializeModel(init_model_args);
  master_model.FitOfflineModel(fit_offline_args);
  // (no assert here - just validate that we didn't crash)
}

// To run this particular test:
// artm_tests.exe --gtest_filter=MasterModel.TestReshapeTokensDecorrelatorPhi
TEST(MasterModel, TestReshapeTokensDecorrelatorPhi) {
  // Purpose of these tests is to validate that phi-regularizers
  // correctly handle n_wt matrix with another layout as compared to p_wt matrix.
  // We randomly change order of words in n_wt matrices,
  // and expect that it does not affect perplexity.
  // This ensures that regularizers act on a correct position in r_wt matrix.

  ::artm::DecorrelatorPhiConfig reg_config;
  reg_config.add_class_id("@default_class");
  testReorderTokens(::artm::RegularizerType_DecorrelatorPhi, reg_config, 10);
}

// To run this particular test:
// artm_tests.exe --gtest_filter=MasterModel.TestReshapeTokensSmoothSparsePhi
TEST(MasterModel, TestReshapeTokensSmoothSparsePhi) {
  ::artm::SmoothSparsePhiConfig reg_config;
  reg_config.add_class_id("@default_class");
  testReorderTokens(::artm::RegularizerType_SmoothSparsePhi, reg_config, 0.1);
}
