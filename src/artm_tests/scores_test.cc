// Copyright 2017, Additive Regularization of Topic Models.

#include <memory>

#include "boost/filesystem.hpp"

#include "gtest/gtest.h"

#include "artm/cpp_interface.h"
#include "artm/core/common.h"
#include "artm/core/instance.h"

#include "artm_tests/test_mother.h"
#include "artm_tests/api.h"

// artm_tests.exe --gtest_filter=Scores.Perplexity
TEST(Scores, Perplexity) {
  int nTokens = 60, nDocs = 10, nTopics = 10;

  ::artm::MasterModelConfig master_config_0 = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  ::artm::test::Helpers::ConfigurePerplexityScore("perplexity_1", &master_config_0, { },
                                                  { artm::core::DefaultTransactionTypeName });
  ::artm::test::Helpers::ConfigurePerplexityScore("perplexity_2", &master_config_0, { });
  master_config_0.add_transaction_typename(artm::core::DefaultTransactionTypeName);
  ::artm::MasterModel master_0(master_config_0);
  ::artm::test::Api api_0(master_0);

  ::artm::MasterModelConfig master_config_1 = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  ::artm::test::Helpers::ConfigurePerplexityScore("perplexity", &master_config_1, { "@error_class" });
  ::artm::MasterModel master_1(master_config_1);
  ::artm::test::Api api_1(master_1);

  ::artm::MasterModelConfig master_config_2 = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  master_config_2.add_class_id("@default_class"); master_config_2.add_class_weight(1.0f);
  master_config_2.add_class_id("@some_class"); master_config_2.add_class_weight(2.0f);
  ::artm::test::Helpers::ConfigurePerplexityScore("perplexity_1", &master_config_2, { });
  ::artm::test::Helpers::ConfigurePerplexityScore("perplexity_2", &master_config_2,
                                                  { "@default_class", "@some_class" });
  ::artm::test::Helpers::ConfigurePerplexityScore("perplexity_3", &master_config_2, { "@default_class" });
  ::artm::test::Helpers::ConfigurePerplexityScore("perplexity_4", &master_config_2,
                                                  { "@error_class", "@some_class" });
  ::artm::MasterModel master_2(master_config_2);
  ::artm::test::Api api_2(master_2);

  ::artm::MasterModelConfig master_config_3 = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  ::artm::test::Helpers::ConfigurePerplexityScore("perplexity", &master_config_3, { });
  ::artm::MasterModel master_3(master_config_3);
  ::artm::test::Api api_3(master_3);

  // Generate doc-token matrix
  artm::Batch batch = ::artm::test::Helpers::GenerateBatch(nTokens, nDocs, "@default_class", "@some_class");
  artm::DictionaryData dict = ::artm::test::Helpers::GenerateDictionary(nTokens, "@default_class", "@some_class");
  std::vector<std::shared_ptr< ::artm::Batch>> batches;
  batches.push_back(std::make_shared< ::artm::Batch>(batch));

  auto offline_args_0 = api_0.Initialize(batches, nullptr, nullptr, &dict);
  auto offline_args_1 = api_1.Initialize(batches, nullptr, nullptr, &dict);
  auto offline_args_2 = api_2.Initialize(batches, nullptr, nullptr, &dict);
  auto offline_args_3 = api_3.Initialize(batches, nullptr, nullptr, &dict);


  master_0.FitOfflineModel(offline_args_0);
  ::artm::GetScoreValueArgs gs;
  gs.set_score_name("perplexity_1");
  auto score = master_0.GetScoreAs< ::artm::PerplexityScore>(gs);
  ASSERT_EQ(score.transaction_typename_info_size(), 1);
  ASSERT_DOUBLE_EQ(score.normalizer(), 0.0);
  ASSERT_DOUBLE_EQ(score.raw(), 0.0);
  ASSERT_EQ(score.zero_words(), 0);
  ASSERT_GT(score.value(), 0.0);
  float value_1 = score.value();

  gs.set_score_name("perplexity_2");
  score = master_0.GetScoreAs< ::artm::PerplexityScore>(gs);
  ASSERT_EQ(score.transaction_typename_info_size(), 1);
  ASSERT_DOUBLE_EQ(score.normalizer(), 0.0);
  ASSERT_DOUBLE_EQ(score.raw(), 0.0);
  ASSERT_EQ(score.zero_words(), 0);
  ASSERT_GT(score.value(), 0.0);
  float value_2 = score.value();

  ASSERT_DOUBLE_EQ(value_1, value_2);


  master_1.FitOfflineModel(offline_args_1);
  gs.set_score_name("perplexity");
  score = master_1.GetScoreAs< ::artm::PerplexityScore>(gs);
  // score calculation should be skipped if class ids sets of model and score have empty intersection
  ASSERT_EQ(score.transaction_typename_info_size(), 0);
  ASSERT_DOUBLE_EQ(score.normalizer(), 0.0);
  ASSERT_DOUBLE_EQ(score.raw(), 0.0);
  ASSERT_DOUBLE_EQ(score.value(), 0.0);
  ASSERT_EQ(score.zero_words(), 0);


  for (int iter = 0; iter < 5; ++iter) {
    master_2.FitOfflineModel(offline_args_2);
    master_3.FitOfflineModel(offline_args_3);
  }

  gs.set_score_name("perplexity_1");
  score = master_2.GetScoreAs< ::artm::PerplexityScore>(gs);
  ASSERT_GT(score.value(), 0.0);
  ASSERT_LT(score.raw(), 0.0);
  ASSERT_GT(score.normalizer(), 0);
  ASSERT_EQ(score.zero_words(), 0);
  ASSERT_EQ(score.transaction_typename_info_size(), 0);
  value_1 = score.value();

  gs.set_score_name("perplexity_2");
  score = master_2.GetScoreAs< ::artm::PerplexityScore>(gs);
  ASSERT_GT(score.value(), 0.0);
  ASSERT_LT(score.raw(), 0.0);
  ASSERT_GT(score.normalizer(), 0);
  ASSERT_EQ(score.zero_words(), 0);
  ASSERT_EQ(score.transaction_typename_info_size(), 0);
  value_2 = score.value();

  ASSERT_DOUBLE_EQ(value_1, value_2);

  gs.set_score_name("perplexity_3");
  score = master_2.GetScoreAs< ::artm::PerplexityScore>(gs);
  ASSERT_GT(score.value(), 0.0);
  ASSERT_LT(score.raw(), 0.0);
  ASSERT_GT(score.normalizer(), 0);
  ASSERT_EQ(score.zero_words(), 0);
  ASSERT_EQ(score.transaction_typename_info_size(), 0);

  gs.set_score_name("perplexity_4");
  score = master_2.GetScoreAs< ::artm::PerplexityScore>(gs);
  ASSERT_GT(score.value(), 0.0);
  ASSERT_LT(score.raw(), 0.0);
  ASSERT_GT(score.normalizer(), 0);
  ASSERT_EQ(score.zero_words(), 0);
  ASSERT_EQ(score.transaction_typename_info_size(), 0);

  gs.set_score_name("perplexity");
  score = master_3.GetScoreAs< ::artm::PerplexityScore>(gs);
  ASSERT_GT(score.value(), 0.0);
  ASSERT_LT(score.raw(), 0.0);
  ASSERT_GT(score.normalizer(), 0.0);
  ASSERT_EQ(score.transaction_typename_info_size(), 0);
}

// artm_tests.exe --gtest_filter=Scores.ScoreTrackerExportImport
TEST(Scores, ScoreTrackerExportImport) {
  int nTokens = 60, nDocs = 10, nTopics = 10, nPasses = 5;

  ::artm::ScoreConfig score_config;
  ::artm::TopTokensScore top_tokens_config;
  score_config.set_config(top_tokens_config.SerializeAsString());
  score_config.set_type(::artm::ScoreType_TopTokens);
  score_config.set_name("top_tokens");

  ::artm::MasterModelConfig master_config_1 = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  ::artm::test::Helpers::ConfigurePerplexityScore("perplexity", &master_config_1);
  master_config_1.add_score_config()->CopyFrom(score_config);
  ::artm::MasterModel master_1(master_config_1);
  ::artm::test::Api api_1(master_1);

  ::artm::MasterModelConfig master_config_2 = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  ::artm::MasterModel master_2(master_config_2);
  ::artm::test::Api api_2(master_2);

  // Generate doc-token matrix
  artm::Batch batch = ::artm::test::Helpers::GenerateBatch(nTokens, nDocs, "@default_class", "@default_class");
  artm::DictionaryData dict = ::artm::test::Helpers::GenerateDictionary(nTokens, "@default_class", "@default_class");
  std::vector<std::shared_ptr< ::artm::Batch>> batches;
  batches.push_back(std::make_shared< ::artm::Batch>(batch));

  auto offline_args = api_1.Initialize(batches, nullptr, nullptr, &dict);
  offline_args.set_num_collection_passes(nPasses);
  master_1.FitOfflineModel(offline_args);

  ::artm::GetScoreArrayArgs args;
  args.set_score_name("perplexity");
  auto score_array_1_perp = master_1.GetScoreArray(args);
  ASSERT_EQ(score_array_1_perp.score_size(), nPasses);

  args.set_score_name("top_tokens");
  auto score_array_1_top = master_1.GetScoreArray(args);
  ASSERT_EQ(score_array_1_top.score_size(), nPasses);

  std::string target_name = artm::test::Helpers::getUniqueString();
  ::artm::ExportScoreTrackerArgs export_args;
  export_args.set_file_name(target_name);
  master_1.ExportScoreTracker(export_args);

  ::artm::ImportScoreTrackerArgs import_args;
  import_args.set_file_name(target_name);
  master_2.ImportScoreTracker(import_args);

  // assert that source and loaded data has same size and content
  args.set_score_name("perplexity");
  auto score_array_2 = master_2.GetScoreArray(args);
  ASSERT_EQ(score_array_2.score_size(), nPasses);
  for (size_t i = 0; i < nPasses; ++i) {
    ASSERT_EQ(score_array_1_perp.score(i).data(), score_array_2.score(i).data());
  }

  args.set_score_name("top_tokens");
  score_array_2 = master_2.GetScoreArray(args);
  ASSERT_EQ(score_array_2.score_size(), nPasses);
  for (size_t i = 0; i < nPasses; ++i) {
    ASSERT_EQ(score_array_1_top.score(i).data(), score_array_2.score(i).data());
  }

  try { boost::filesystem::remove(target_name); }
  catch (...) { }
}
