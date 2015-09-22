// Copyright 2014, Additive Regularization of Topic Models.

#include "boost/thread.hpp"
#include "gtest/gtest.h"

#include "boost/filesystem.hpp"

#include "artm/cpp_interface.h"
#include "artm/core/exceptions.h"
#include "artm/messages.pb.h"

#include "artm/core/internals.pb.h"
#include "artm/core/helpers.h"

#include "artm_tests/test_mother.h"

namespace fs = boost::filesystem;

TEST(CppInterface, Canary) {
}

void BasicTest() {
  std::string target_path = artm::test::Helpers::getUniqueString();
  const int nTopics = 5;

  ::artm::MasterComponentConfig master_config;
  master_config.set_cache_theta(true);

  ::artm::ScoreConfig score_config;
  score_config.set_config(::artm::PerplexityScoreConfig().SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_Perplexity);
  score_config.set_name("PerplexityScore");
  master_config.add_score_config()->CopyFrom(score_config);
  master_config.set_disk_cache_path(".");

  // Create master component
  std::unique_ptr<artm::MasterComponent> master_component;
  master_component.reset(new ::artm::MasterComponent(master_config));
  EXPECT_EQ(master_component->info()->score_size(), 1);

  // Create regularizers
  std::string reg_decor_name = "decorrelator";
  artm::DecorrelatorPhiConfig decor_config;
  artm::RegularizerConfig reg_decor_config;
  reg_decor_config.set_name(reg_decor_name);
  reg_decor_config.set_type(artm::RegularizerConfig_Type_DecorrelatorPhi);
  reg_decor_config.set_config(decor_config.SerializeAsString());
  std::shared_ptr<artm::Regularizer> decorrelator_reg(
    new artm::Regularizer(*(master_component.get()), reg_decor_config));

  std::string reg_multilang_name = "multilanguage";
  artm::MultiLanguagePhiConfig multilang_config;
  artm::RegularizerConfig reg_multilang_config;
  reg_multilang_config.set_name(reg_multilang_name);
  reg_multilang_config.set_type(artm::RegularizerConfig_Type_MultiLanguagePhi);
  reg_multilang_config.set_config(multilang_config.SerializeAsString());
  std::shared_ptr<artm::Regularizer> multilanguage_reg(
    new artm::Regularizer(*(master_component.get()), reg_multilang_config));

  EXPECT_EQ(master_component->info()->regularizer_size(), 2);

  // Create model
  artm::ModelConfig model_config;
  model_config.set_topics_count(nTopics);
  model_config.add_topic_name("first topic");
  model_config.add_topic_name("second topic");
  model_config.add_topic_name("third topic");
  model_config.add_topic_name("4th topic");
  model_config.add_topic_name("5th topic");
  EXPECT_EQ(model_config.topic_name_size(), nTopics);
  model_config.add_regularizer_name(reg_decor_name);
  model_config.add_regularizer_tau(1);
  model_config.add_regularizer_name(reg_multilang_name);
  model_config.add_regularizer_tau(1);
  model_config.set_name("model_config1");
  model_config.set_use_ptdw_matrix(true);  // temporary switch tests into use_ptdw_matrix mode
  artm::Model model(*master_component, model_config);

  // Load doc-token matrix
  int nTokens = 10;
  int nDocs = 15;

  artm::Batch batch;
  batch.set_id("00b6d631-46a6-4edf-8ef6-016c7b27d9f0");
  for (int i = 0; i < nTokens; i++) {
    std::stringstream str;
    str << "token" << i;
    batch.add_token(str.str());
  }

  std::vector<std::string> item_title;
  for (int iDoc = 0; iDoc < nDocs; iDoc++) {
    artm::Item* item = batch.add_item();
    std::stringstream str;
    str << "item" << iDoc;
    item_title.push_back(str.str());
    item->set_title(str.str());
    item->set_id(666 + iDoc);
    artm::Field* field = item->add_field();
    for (int iToken = 0; iToken < nTokens; ++iToken) {
      field->add_token_id(iToken);
      field->add_token_weight(static_cast<float>(iDoc + iToken + 1));
    }
  }

  EXPECT_EQ(batch.item().size(), nDocs);
  for (int i = 0; i < batch.item().size(); i++) {
    EXPECT_EQ(batch.item().Get(i).field().Get(0).token_id().size(),
        nTokens);
  }

  // Index doc-token matrix

  std::shared_ptr<artm::TopicModel> topic_model;
  double expected_normalizer = 0;
  double previous_perplexity = 0;
  for (int iter = 0; iter < 5; ++iter) {
    master_component->AddBatch(batch, /*reset_scores =*/ true);  // NOLINT
    master_component->WaitIdle();
    model.Synchronize(0.0);
    {
      artm::GetTopicModelArgs args;
      args.set_model_name(model.name());
      for (int i = 0; i < nTopics; ++i) {
        args.add_topic_name(model_config.topic_name(i));
      }
      for (int i = 0; i < nTokens; i++) {
        args.add_token("token" + std::to_string(i));
        args.add_class_id("@default_class");
      }

      topic_model = master_component->GetTopicModel(args);
    }

    std::shared_ptr< ::artm::PerplexityScore> perplexity =
      master_component->GetScoreAs< ::artm::PerplexityScore>(model, "PerplexityScore");

    if (iter > 0)
      EXPECT_EQ(perplexity->value(), previous_perplexity);

    artm::GetScoreValueArgs score_args;
    score_args.set_model_name(model.name());
    score_args.set_score_name("PerplexityScore");
    score_args.mutable_batch()->CopyFrom(batch);
    auto perplexity_data = master_component->GetScore(score_args);
    auto perplexity2 = std::make_shared< ::artm::PerplexityScore>();
    perplexity2->ParseFromString(perplexity_data->data());
    previous_perplexity = perplexity2->value();

    if (iter == 1) {
      expected_normalizer = perplexity->normalizer();
      EXPECT_GT(expected_normalizer, 0);

      try {
      master_component->GetRegularizerState(reg_decor_name);
      EXPECT_FALSE(true);
      } catch (std::runtime_error& err_obj) {
        std::cout << err_obj.what() << std::endl;
        EXPECT_TRUE(true);
      }

    } else if (iter >= 2) {
      // Verify that normalizer does not grow starting from second iteration.
      // This confirms that the Instance::ForceResetScores() function works as expected.
      EXPECT_EQ(perplexity->normalizer(), expected_normalizer);
    }
  }

  master_component->AddBatch(batch, /*reset_scores =*/ true);  // NOLINT

  EXPECT_TRUE(master_component->WaitIdle());

  auto old_state_wrapper = master_component->GetRegularizerState(reg_multilang_name);
  model.Synchronize(1.0);
  auto new_state_wrapper = master_component->GetRegularizerState(reg_multilang_name);

  artm::MultiLanguagePhiInternalState old_state;
  artm::MultiLanguagePhiInternalState new_state;
  old_state.ParseFromString(old_state_wrapper->data());
  new_state.ParseFromString(new_state_wrapper->data());

  int saved_value = new_state.no_regularization_calls();
  EXPECT_EQ(saved_value - old_state.no_regularization_calls(), 1);

  multilanguage_reg->Reconfigure(reg_multilang_config);
  new_state_wrapper = master_component->GetRegularizerState(reg_multilang_name);
  new_state.ParseFromString(new_state_wrapper->data());
  EXPECT_EQ(new_state.no_regularization_calls(), saved_value);


  model.Disable();

  int nUniqueTokens = nTokens;
  ASSERT_EQ(nUniqueTokens, topic_model->token_size());
  auto first_token_topics = topic_model->token_weights(0);
  EXPECT_EQ(first_token_topics.value_size(), nTopics);

  {
    artm::GetThetaMatrixArgs args;
    args.set_model_name(model.name().c_str());
    std::shared_ptr< ::artm::ThetaMatrix> theta_matrix = master_component->GetThetaMatrix(args);

    EXPECT_EQ(theta_matrix->item_id_size(), nDocs);
    EXPECT_EQ(theta_matrix->item_title_size(), nDocs);
    EXPECT_EQ(theta_matrix->topics_count(), nTopics);
    for (int item_index = 0; item_index < theta_matrix->item_id_size(); ++item_index) {
      EXPECT_EQ(theta_matrix->item_id(item_index), 666 + item_index);
      EXPECT_EQ(theta_matrix->item_title(item_index), item_title[item_index]);
      const ::artm::FloatArray& weights = theta_matrix->item_weights(item_index);
      ASSERT_EQ(weights.value_size(), nTopics);
      float sum = 0;
      for (int topic_index = 0; topic_index < weights.value_size(); ++topic_index) {
        float weight = weights.value(topic_index);
        EXPECT_GT(weight, 0);
        sum += weight;
      }

      EXPECT_LE(abs(sum - 1), 0.001);
    }

    args.add_topic_index(2); args.add_topic_index(3);  // retrieve 2nd and 3rd topic
    std::shared_ptr< ::artm::ThetaMatrix> theta_matrix23 = master_component->GetThetaMatrix(args);
    EXPECT_EQ(theta_matrix23->item_id_size(), nDocs);
    EXPECT_EQ(theta_matrix23->topics_count(), 2);
    for (int item_index = 0; item_index < theta_matrix23->item_id_size(); ++item_index) {
      const ::artm::FloatArray& weights23 = theta_matrix23->item_weights(item_index);
      const ::artm::FloatArray& weights = theta_matrix->item_weights(item_index);
      ASSERT_EQ(weights23.value_size(), 2);
      EXPECT_EQ(weights23.value(0), weights.value(2));
      EXPECT_EQ(weights23.value(1), weights.value(3));
    }

    args.clear_topic_index();
    args.add_topic_name(topic_model->topic_name(2));  // retrieve 2nd and 3rd topic (but use topic_names)
    args.add_topic_name(topic_model->topic_name(3));
    theta_matrix23 = master_component->GetThetaMatrix(args);
    EXPECT_EQ(theta_matrix23->topic_name_size(), 2);
    EXPECT_EQ(theta_matrix23->topic_name(0), topic_model->topic_name(2));
    EXPECT_EQ(theta_matrix23->topic_name(1), topic_model->topic_name(3));
    EXPECT_EQ(theta_matrix23->item_id_size(), nDocs);
    for (int item_index = 0; item_index < theta_matrix23->item_id_size(); ++item_index) {
      const ::artm::FloatArray& weights23 = theta_matrix23->item_weights(item_index);
      const ::artm::FloatArray& weights = theta_matrix->item_weights(item_index);
      ASSERT_EQ(weights23.value_size(), 2);
      EXPECT_EQ(weights23.value(0), weights.value(2));
      EXPECT_EQ(weights23.value(1), weights.value(3));
    }


    args.clear_topic_name();
    args.mutable_batch()->CopyFrom(batch);
    std::shared_ptr< ::artm::ThetaMatrix> theta_matrix2 = master_component->GetThetaMatrix(args);
    EXPECT_EQ(theta_matrix2->item_id_size(), nDocs);
    EXPECT_EQ(theta_matrix2->item_title_size(), nDocs);
    EXPECT_EQ(theta_matrix2->topics_count(), nTopics);
    for (int item_index = 0; item_index < theta_matrix2->item_id_size(); ++item_index) {
      EXPECT_EQ(theta_matrix2->item_id(item_index), 666 + item_index);
      EXPECT_EQ(theta_matrix2->item_title(item_index), item_title[item_index]);
      const ::artm::FloatArray& weights2 = theta_matrix2->item_weights(item_index);
      EXPECT_EQ(weights2.value_size(), nTopics);
      float sum2 = 0;
      for (int topic_index = 0; topic_index < weights2.value_size(); ++topic_index) {
        float weight2 = weights2.value(topic_index);
        EXPECT_GT(weight2, 0);
        sum2 += weight2;
      }

      EXPECT_LE(abs(sum2 - 1), 0.001);
    }
  }

  model_config.set_name("model2_name");
  artm::Model model2(*master_component, model_config);
  // Test overwrite topic model
  artm::TopicModel new_topic_model;
  new_topic_model.set_name(model2.name());
  new_topic_model.mutable_topic_name()->CopyFrom(model_config.topic_name());
  new_topic_model.add_token("my overwritten token");
  new_topic_model.add_token("my overwritten token2");
  new_topic_model.add_operation_type(::artm::TopicModel_OperationType_Increment);
  new_topic_model.add_operation_type(::artm::TopicModel_OperationType_Increment);
  auto weights = new_topic_model.add_token_weights();
  auto weights2 = new_topic_model.add_token_weights();
  for (int i = 0; i < nTopics; ++i) {
    weights->add_value(static_cast<float>(i));
    weights2->add_value(static_cast<float>(nTopics - i));
  }

  model2.Overwrite(new_topic_model);

  {
    artm::GetTopicModelArgs args;
    args.set_model_name(model2.name());
    for (int i = 0; i < nTopics; ++i) {
      args.add_topic_name(model_config.topic_name(i));
    }
    args.add_token("my overwritten token");
    args.add_class_id("@default_class");
    args.add_token("my overwritten token2");
    args.add_class_id("@default_class");

    auto new_topic_model2 = master_component->GetTopicModel(args);
    ASSERT_EQ(new_topic_model2->token_size(), 2);
    EXPECT_EQ(new_topic_model2->token(0), "my overwritten token");
    EXPECT_EQ(new_topic_model2->token(1), "my overwritten token2");
    for (int i = 0; i < nTopics; ++i) {
      EXPECT_FLOAT_EQ(
        new_topic_model2->token_weights(0).value(i),
        static_cast<float>(i) / static_cast<float>(nTopics));

      EXPECT_FLOAT_EQ(
        new_topic_model2->token_weights(1).value(i),
        1.0f - static_cast<float>(i) / static_cast<float>(nTopics));
    }
  }

  // Test dictionaries and InitializeModel
  ::artm::DictionaryConfig dict_config;
  dict_config.set_name("My dictionary");
  ::artm::DictionaryEntry* de1 = dict_config.add_entry();
  ::artm::DictionaryEntry* de2 = dict_config.add_entry();
  ::artm::DictionaryEntry* de3 = dict_config.add_entry();
  de1->set_key_token("my_tok_1");
  de2->set_key_token("my_tok_2");
  de3->set_key_token("my_tok_3");
  ::artm::Dictionary dict(*master_component, dict_config);
  model_config.set_name("model3_name");
  artm::Model model3(*master_component, model_config);
  model3.Initialize(dict);

  artm::GetTopicModelArgs args;
  args.set_model_name(model3.name());
  for (int i = 0; i < nTopics; ++i) {
    args.add_topic_name(model_config.topic_name(i));
  }
  for (int i = 0; i < nTokens; i++) {
    args.add_token("my_tok_" + std::to_string(i));
    args.add_class_id("@default_class");
  }

  auto new_topic_model3 = master_component->GetTopicModel(args);
  ASSERT_EQ(new_topic_model3->token_size(), 3);
  ASSERT_EQ(new_topic_model3->token(0), "my_tok_1");
  ASSERT_EQ(new_topic_model3->token(1), "my_tok_2");
  ASSERT_EQ(new_topic_model3->token(2), "my_tok_3");

  artm::ModelConfig model_config2(model_config);
  model_config2.clear_topic_name();
  model_config2.add_topic_name(model_config.topic_name(0));  // todo(alfrey) - remove this line
  model_config2.add_topic_name(model_config.topic_name(1));
  model_config2.add_topic_name(model_config.topic_name(2));
  model_config2.add_topic_name(model_config.topic_name(3));
  model_config2.add_topic_name(model_config.topic_name(4));
  model3.Reconfigure(model_config2);

  model3.Synchronize(0.0);
  args.Clear();
  args.set_model_name(model_config2.name());
  auto new_topic_model4 = master_component->GetTopicModel(args);

  // ToDo(alfrey): uncomment this asserts
  // ASSERT_EQ(new_topic_model4->topic_name_size(), 4);
  // EXPECT_EQ(new_topic_model4->topic_name(0), model_config.topic_name(1));
  // EXPECT_EQ(new_topic_model4->topic_name(1), model_config.topic_name(2));
  // EXPECT_EQ(new_topic_model4->topic_name(2), model_config.topic_name(3));
  // EXPECT_EQ(new_topic_model4->topic_name(3), model_config.topic_name(4));

  master_component.reset();

  try { boost::filesystem::remove_all(target_path); }
  catch (...) {}
}

// artm_tests.exe --gtest_filter=CppInterface.BasicTest_StandaloneMode
TEST(CppInterface, BasicTest_StandaloneMode) {
  BasicTest();
}

// artm_tests.exe --gtest_filter=CppInterface.ModelExceptions
TEST(CppInterface, ModelExceptions) {
  // Create instance
  artm::MasterComponentConfig master_config;
  artm::MasterComponent master_component(master_config);

  // Create model
  artm::ModelConfig model_config;
  model_config.set_topics_count(10);
  model_config.set_name("model_config1");
  artm::Model model(master_component, model_config);
}

// artm_tests.exe --gtest_filter=CppInterface.WaitIdleTimeout
TEST(CppInterface, WaitIdleTimeout) {
  ::artm::MasterComponentConfig master_config;
  master_config.set_processor_queue_max_size(10000);
  master_config.set_merger_queue_max_size(10000);
  ::artm::MasterComponent master(master_config);
  ::artm::ModelConfig model_config;
  model_config.set_name("model_config1");
  model_config.set_inner_iterations_count(100000);

  ::artm::Model model(master, model_config);
  ::artm::Batch batch;
  batch.set_id("00b6d631-46a6-4edf-8ef6-016c7b27d9f0");
  for (int i = 0; i < 10; ++i) {
    ::artm::Item* item = batch.add_item();
    ::artm::Field* field = item->add_field();
    field->add_token_id(i);
    field->add_token_weight(static_cast<float>(i + 1));
    batch.add_token(artm::test::Helpers::getUniqueString());
  }

  master.AddBatch(batch);
  EXPECT_TRUE(master.WaitIdle());
  model.Synchronize(0.0);

  master.AddBatch(batch);
  EXPECT_FALSE(master.WaitIdle(0));
}

// artm_tests.exe --gtest_filter=CppInterface.GatherNewTokens
TEST(CppInterface, GatherNewTokens) {
  artm::MasterComponentConfig master_config;
  artm::MasterComponent master(master_config);

  artm::ModelConfig model_config;
  model_config.set_topics_count(10);
  model_config.set_name("model_config1");
  artm::Model model(master, model_config);

  std::string token1 = artm::test::Helpers::getUniqueString();
  std::string token2 = artm::test::Helpers::getUniqueString();

  // Generate batch with one token (token1)
  ::artm::Batch batch;
  batch.set_id(artm::test::Helpers::getUniqueString());
  batch.add_token(token1);
  ::artm::Item* item = batch.add_item();
  ::artm::Field* field = item->add_field();
  field->add_token_id(0);
  field->add_token_weight(1.0f);

  // Process batch and expect that token is automatically picked up by the model
  master.AddBatch(batch);
  master.WaitIdle();
  model.Synchronize(1.0);
  auto tm1 = master.GetTopicModel(model.name());
  ASSERT_EQ(tm1->token_size(), 1);
  ASSERT_EQ(tm1->token(0), token1);

  // Change configuration to not use new tokens
  model_config.set_use_new_tokens(false);
  model.Reconfigure(model_config);

  // Create different batch that contains token2
  batch.mutable_token(0)->assign(token2);

  // Process batch with token2, and expect that it is ignored by the model
  master.AddBatch(batch);
  master.WaitIdle();
  model.Synchronize(1.0);
  auto tm2 = master.GetTopicModel(model.name());
  ASSERT_EQ(tm2->token_size(), 1);  // new token is ignored
  ASSERT_EQ(tm2->token(0), token1);

  // Change configuration back to use new tokens
  model_config.set_use_new_tokens(true);
  model.Reconfigure(model_config);
  master.AddBatch(batch);
  master.WaitIdle();
  model.Synchronize(1.0);
  auto tm3 = master.GetTopicModel(model.name());
  ASSERT_EQ(tm3->token_size(), 2);  // now new token is picked up
  ASSERT_TRUE((tm3->token(0) == token1 && tm3->token(1) == token2) ||
              (tm3->token(0) == token2 && tm3->token(1) == token1));
}

// artm_tests.exe --gtest_filter=CppInterface.ProcessBatchesApi
TEST(CppInterface, ProcessBatchesApi) {
  int nTopics = 17;
  int nBatches = 5;

  std::string target_folder = artm::test::Helpers::getUniqueString();
  ::artm::test::TestMother::GenerateBatches(nBatches, 50, target_folder);

  artm::MasterComponentConfig master_config;
  master_config.set_disk_path(target_folder);
  artm::ScoreConfig* score_config = master_config.add_score_config();
  score_config->set_name("Perplexity");
  score_config->set_type(artm::ScoreConfig_Type_Perplexity);
  ::artm::PerplexityScoreConfig perplexity_score_config;
  score_config->set_config(perplexity_score_config.SerializeAsString());
  artm::MasterComponent master(master_config);

  std::vector<std::string> all_batches = ::artm::core::BatchHelpers::ListAllBatches(target_folder);
  ASSERT_EQ(all_batches.size(), nBatches);

  artm::ImportBatchesArgs import_batches_args;
  for (auto& batch_path : all_batches) {
    std::shared_ptr< ::artm::Batch> batch = artm::LoadBatch(batch_path);
    import_batches_args.add_batch()->CopyFrom(*batch);
    import_batches_args.add_batch_name(batch->id());
  }
  master.ImportBatches(import_batches_args);

  ::artm::ModelConfig model_config;
  model_config.set_name("pwt0");
  model_config.set_topics_count(nTopics);
  ::artm::Model model(master, model_config);

  artm::InitializeModelArgs initialize_model_args;
  for (auto& batch_name : import_batches_args.batch_name())
    initialize_model_args.add_batch_filename(batch_name);
  initialize_model_args.set_source_type(artm::InitializeModelArgs_SourceType_Batches);
  initialize_model_args.set_topics_count(nTopics);
  initialize_model_args.set_model_name("pwt0");
  master.InitializeModel(initialize_model_args);
  std::shared_ptr< ::artm::MasterComponentInfo> master_info = master.info();
  ASSERT_EQ(master_info->model_size(), 1);  // "pwt0"
  EXPECT_EQ(master_info->model(0).name(), "pwt0");
  EXPECT_EQ(master_info->model(0).topics_count(), nTopics);

  std::shared_ptr< ::artm::TopicModel> pwt_model = master.GetTopicModel("pwt0");
  ASSERT_NE(pwt_model, nullptr);
  ASSERT_EQ(pwt_model->topics_count(), nTopics);

  // Test export and import of new-style models
  artm::ExportModelArgs export_model_args;
  export_model_args.set_model_name(pwt_model->name());

  fs::path export_filename = (fs::path(target_folder) / fs::path(artm::test::Helpers::getUniqueString() + ".model"));
  export_model_args.set_model_name("pwt0");
  export_model_args.set_file_name(export_filename.string());
  artm::ImportModelArgs import_model_args;
  import_model_args.set_model_name("import_pwt");
  import_model_args.set_file_name(export_model_args.file_name());

  master.ExportModel(export_model_args);
  master.ImportModel(import_model_args);
  ASSERT_EQ(master.info()->model_size(), 2);  // "pwt0", "import_pwt"
  bool ok2 = false;
  ::artm::test::Helpers::CompareTopicModels(*master.GetTopicModel("pwt0"), *master.GetTopicModel("import_pwt"), &ok2);
  if (!ok2) {
    std::cout << "Exported topic model:\n"
      << ::artm::test::Helpers::DescribeTopicModel(*master.GetTopicModel("pwt0"));
    std::cout << "Imported topic model:\n"
      << ::artm::test::Helpers::DescribeTopicModel(*master.GetTopicModel("import_pwt"));
  }

  master.DisposeModel("import_pwt");
  ASSERT_EQ(master.info()->model_size(), 1);  // "pwt0"
  /////////////////////////////////////////////

  artm::ProcessBatchesArgs process_batches_args;
  for (auto& batch_name : import_batches_args.batch_name())
    process_batches_args.add_batch_filename(batch_name);
  process_batches_args.set_nwt_target_name("nwt_hat");

  artm::NormalizeModelArgs normalize_model_args;
  normalize_model_args.set_pwt_target_name("pwt");
  normalize_model_args.set_nwt_source_name("nwt_hat");

  std::shared_ptr< ::artm::PerplexityScore> perplexity_score;
  std::shared_ptr< ::artm::Matrix> attached_phi;
  for (int i = 0; i < 10; ++i) {  // 10 iterations
    process_batches_args.set_pwt_source_name(i == 0 ? "pwt0" : "pwt");
    process_batches_args.set_theta_matrix_type(artm::ProcessBatchesArgs_ThetaMatrixType_Dense);
    std::shared_ptr< ::artm::ProcessBatchesResultObject> result = master.ProcessBatches(process_batches_args);
    perplexity_score = result->GetScoreAs< ::artm::PerplexityScore>("Perplexity");
    EXPECT_EQ(result->GetThetaMatrix().topics_count(), nTopics);
    EXPECT_EQ(result->GetThetaMatrix().item_id_size(), nBatches);  // assuming that each batch has just one document
    master.NormalizeModel(normalize_model_args);
  }

  ASSERT_EQ(master.info()->model_size(), 3);  // "pwt0", "pwt", "nwt_hat"

  EXPECT_NE(perplexity_score, nullptr);
  EXPECT_NE(perplexity_score->value(), 0.0);

  for (int i = 0; i < 10; ++i) {
    master.InvokeIteration();
    master.WaitIdle();
    model.Synchronize(0.0);
  }

  auto perplexity_score2 = master.GetScoreAs< ::artm::PerplexityScore>(model, "Perplexity");
  EXPECT_NE(perplexity_score2, nullptr);
  ASSERT_APPROX_EQ(perplexity_score2->value(), perplexity_score->value());

  bool ok = false;
  ::artm::test::Helpers::CompareTopicModels(*master.GetTopicModel("pwt"), *master.GetTopicModel("pwt0"), &ok);

  if (!ok) {
    std::cout << "New-tuned topic model:\n"
      << ::artm::test::Helpers::DescribeTopicModel(*master.GetTopicModel("pwt"));
    std::cout << "Old-tuned topic model:\n"
      << ::artm::test::Helpers::DescribeTopicModel(*master.GetTopicModel("pwt0"));
  }

  ::artm::DictionaryConfig dict_config;
  dict_config.set_name("My dictionary");
  ::artm::DictionaryEntry* de1 = dict_config.add_entry();
  de1->set_key_token("my_tok_1");
  ::artm::Dictionary dict(master, dict_config);
  master_info = master.info();
  ASSERT_EQ(master_info->dictionary_size(), 1);
  EXPECT_EQ(master_info->dictionary(0).entries_count(), 1);

  {
    artm::MasterComponent master_clone(master);
    std::shared_ptr< ::artm::MasterComponentInfo> clone_master_info = master_clone.info();
    ASSERT_EQ(clone_master_info->model_size(), 2);  // "pwt", "nwt_hat"; "pwt0" is not cloned (old-style model)
    ASSERT_EQ(clone_master_info->dictionary_size(), 1);
    EXPECT_EQ(clone_master_info->dictionary(0).entries_count(), 1);
    ASSERT_EQ(clone_master_info->score_size(), master_info->score_size());
    ASSERT_EQ(clone_master_info->regularizer_size(), master_info->regularizer_size());

    ::artm::test::Helpers::CompareTopicModels(*master_clone.GetTopicModel("pwt"),
                                              *master.GetTopicModel("pwt"), &ok);
    ASSERT_TRUE(ok);
  }

  // Verify that we may call ProcessBatches without nwt_target
  process_batches_args.clear_nwt_target_name();
  std::shared_ptr< ::artm::ProcessBatchesResultObject> result_1 = master.ProcessBatches(process_batches_args);
  perplexity_score = result_1->GetScoreAs< ::artm::PerplexityScore>("Perplexity");
  EXPECT_NE(perplexity_score, nullptr);
  EXPECT_NE(perplexity_score->value(), 0.0);

  // Dummy test to verify we can merge models
  ::artm::MergeModelArgs merge_model_args;
  merge_model_args.add_nwt_source_name("pwt"); merge_model_args.add_source_weight(1.0f);
  merge_model_args.add_nwt_source_name("pwt0"); merge_model_args.add_source_weight(1.0f);
  merge_model_args.set_nwt_target_name("nwt_merge");
  master.MergeModel(merge_model_args);
  std::shared_ptr< ::artm::TopicModel> nwt_merge = master.GetTopicModel("nwt_merge");
  ASSERT_NE(nwt_merge, nullptr);
  ASSERT_EQ(nwt_merge->topics_count(), nTopics);

  // Dummy test to verify we can regularize models
  ::artm::RegularizerConfig sparse_phi_config;
  sparse_phi_config.set_name("sparse_phi");
  sparse_phi_config.set_type(::artm::RegularizerConfig_Type_SmoothSparsePhi);
  sparse_phi_config.set_config(::artm::SmoothSparsePhiConfig().SerializeAsString());
  ::artm::Regularizer sparse_phi(master, sparse_phi_config);

  ::artm::RegularizeModelArgs regularize_model_args;
  regularize_model_args.set_rwt_target_name("rwt");
  regularize_model_args.set_pwt_source_name("pwt");
  regularize_model_args.set_nwt_source_name("nwt_hat");
  ::artm::RegularizerSettings* regularizer_settings = regularize_model_args.add_regularizer_settings();
  regularizer_settings->set_name("sparse_phi");
  regularizer_settings->set_tau(-0.5);
  master.RegularizeModel(regularize_model_args);
  std::shared_ptr< ::artm::TopicModel> rwt = master.GetTopicModel("rwt");
  ASSERT_NE(rwt, nullptr);
  ASSERT_EQ(rwt->topics_count(), nTopics);

  // Test to verify Ptdw extraction
  process_batches_args.set_use_ptdw_matrix(true);
  process_batches_args.set_theta_matrix_type(artm::ProcessBatchesArgs_ThetaMatrixType_Ptdw);
  std::shared_ptr< ::artm::ProcessBatchesResultObject> result_2 = master.ProcessBatches(process_batches_args);
  auto& theta_matrix = result_2->GetThetaMatrix();
  ASSERT_EQ(theta_matrix.item_id_size(), 79);
  ASSERT_EQ(theta_matrix.topic_index_size(), 79);
  ASSERT_EQ(theta_matrix.item_weights_size(), 79);
  for (int index = 0; index < theta_matrix.item_id_size(); ++index) {
    ASSERT_GE(theta_matrix.topic_index(index).value_size(), 0);
    ASSERT_GE(theta_matrix.item_weights(index).value_size(), 0);
    ASSERT_EQ(theta_matrix.topic_index(index).value_size(), theta_matrix.item_weights(index).value_size());
  }

  try { boost::filesystem::remove_all(target_folder); }
  catch (...) {}
}

// artm_tests.exe --gtest_filter=CppInterface.AttachModel
TEST(CppInterface, AttachModel) {
  int nTopics = 17, nBatches = 5;
  std::string target_folder = artm::test::Helpers::getUniqueString();
  ::artm::test::TestMother::GenerateBatches(nBatches, 50, target_folder);
  artm::MasterComponentConfig master_config;
  artm::MasterComponent master(master_config);

  // Verify that it is possible to attach immediatelly after Initialize()
  artm::InitializeModelArgs initialize_model_args;
  initialize_model_args.set_disk_path(target_folder);
  initialize_model_args.set_source_type(artm::InitializeModelArgs_SourceType_Batches);
  initialize_model_args.set_topics_count(nTopics);
  initialize_model_args.set_model_name("pwt0");
  master.InitializeModel(initialize_model_args);
  std::shared_ptr< ::artm::Matrix> attached_pwt = master.AttachTopicModel("pwt0");
  std::shared_ptr< ::artm::TopicModel> pwt0_model = master.GetTopicModel("pwt0");
  ASSERT_EQ(attached_pwt->no_rows(), pwt0_model->token_size());
  ASSERT_EQ(attached_pwt->no_columns(), pwt0_model->topics_count());

  ::artm::MergeModelArgs merge_model_args;
  merge_model_args.add_nwt_source_name("pwt0"); merge_model_args.add_source_weight(1.0f);
  merge_model_args.set_nwt_target_name("nwt_merge");
  master.MergeModel(merge_model_args);
  std::shared_ptr< ::artm::Matrix> attached_nwt_merge = master.AttachTopicModel("nwt_merge");
  std::shared_ptr< ::artm::TopicModel> nwt_merge_model = master.GetTopicModel("nwt_merge");
  ASSERT_EQ(attached_nwt_merge->no_rows(), nwt_merge_model->token_size());
  ASSERT_EQ(attached_nwt_merge->no_columns(), nwt_merge_model->topics_count());

  // Verify that it is possible to modify the attached matrix
  for (int token_index = 0; token_index < nwt_merge_model->token_size(); ++token_index) {
    for (int topic_index = 0; topic_index < nwt_merge_model->topics_count(); ++topic_index) {
      EXPECT_EQ((*attached_nwt_merge)(token_index, topic_index),
                nwt_merge_model->token_weights(token_index).value(topic_index));
      (*attached_nwt_merge)(token_index, topic_index) = 2.0f * token_index + 3.0f * topic_index;
    }
  }

  std::shared_ptr< ::artm::TopicModel> updated_model = master.GetTopicModel("nwt_merge");
  for (int token_index = 0; token_index < nwt_merge_model->token_size(); ++token_index) {
    for (int topic_index = 0; topic_index < nwt_merge_model->topics_count(); ++topic_index) {
      EXPECT_EQ(updated_model->token_weights(token_index).value(topic_index),
                2.0f * token_index + 3.0f * topic_index);
    }
  }

  {
    bool ok = false;
    artm::MasterComponent master_clone(master);
    ::artm::test::Helpers::CompareTopicModels(*master_clone.GetTopicModel("nwt_merge"),
      *master.GetTopicModel("nwt_merge"), &ok);
    ASSERT_TRUE(ok);
  }

  // Good practice is to dispose model once its attachment is gone.
  master.DisposeModel("pwt");
  master.DisposeModel("nwt_merge");


  try { boost::filesystem::remove_all(target_folder); }
  catch (...) {}
}
