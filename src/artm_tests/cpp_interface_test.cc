// Copyright 2014, Additive Regularization of Topic Models.

#include "boost/thread.hpp"
#include "gtest/gtest.h"

#include "boost/filesystem.hpp"

#include "artm/cpp_interface.h"
#include "artm/core/exceptions.h"
#include "artm/messages.pb.h"

#include "artm/core/internals.pb.h"

TEST(CppInterface, Canary) {
}

void BasicTest(bool is_network_mode, bool is_proxy_mode) {
  const int nTopics = 5;

  // Endpoints:
  // 5555 - master component (network_mode)
  // 5556 - node controller for workers (network_mode)
  // 5557 - node controller for master (proxy_mode)

  std::shared_ptr< ::artm::NodeController> node_controller;
  std::shared_ptr< ::artm::NodeController> node_controller_master;
  ::artm::MasterComponentConfig master_config;
  if (is_network_mode) {
    ::artm::NodeControllerConfig node_config;
    node_config.set_create_endpoint("tcp://*:5556");
    node_controller.reset(new ::artm::NodeController(node_config));

    master_config.set_create_endpoint("tcp://*:5555");
    master_config.set_connect_endpoint("tcp://localhost:5555");
    master_config.add_node_connect_endpoint("tcp://localhost:5556");
    master_config.set_disk_path(".");

    // Clean all .batches files
    boost::filesystem::recursive_directory_iterator it(".");
    boost::filesystem::recursive_directory_iterator endit;
    while (it != endit) {
      if (boost::filesystem::is_regular_file(*it) && it->path().extension() == ".batch") {
        boost::filesystem::remove(*it);
      }

      ++it;
    }

    master_config.set_modus_operandi(::artm::MasterComponentConfig_ModusOperandi_Network);
  } else {
    master_config.set_modus_operandi(::artm::MasterComponentConfig_ModusOperandi_Local);
    master_config.set_cache_theta(true);
  }

  ::artm::ScoreConfig score_config;
  score_config.set_config(::artm::PerplexityScoreConfig().SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_Perplexity);
  score_config.set_name("PerplexityScore");
  master_config.add_score_config()->CopyFrom(score_config);
  master_config.set_disk_cache_path(".");

  // Create master component
  std::unique_ptr<artm::MasterComponent> master_component;
  if (!is_proxy_mode) {
    master_component.reset(new ::artm::MasterComponent(master_config));
  } else {
    ::artm::NodeControllerConfig node_config;
    node_config.set_create_endpoint("tcp://*:5557");
    node_controller_master.reset(new ::artm::NodeController(node_config));

    ::artm::MasterProxyConfig master_proxy_config;
    master_proxy_config.mutable_config()->CopyFrom(master_config);
    master_proxy_config.set_node_connect_endpoint("tcp://localhost:5557");
    master_component.reset(new ::artm::MasterComponent(master_proxy_config));
  }

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

  // Create model
  artm::ModelConfig model_config;
  model_config.set_topics_count(nTopics);
  model_config.add_topic_name("first topic");
  model_config.add_topic_name("second topic");
  model_config.add_topic_name("third topic");
  model_config.add_topic_name("4th topic");
  model_config.add_topic_name("5th topic");
  EXPECT_EQ(model_config.topic_name_size(), nTopics);
  model_config.add_score_name("PerplexityScore");
  model_config.add_regularizer_name(reg_decor_name);
  model_config.add_regularizer_tau(1);
  model_config.add_regularizer_name(reg_multilang_name);
  model_config.add_regularizer_tau(1);
  model_config.set_name("model_config1");
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
      field->add_token_count(iDoc + iToken + 1);
    }
  }

  EXPECT_EQ(batch.item().size(), nDocs);
  for (int i = 0; i < batch.item().size(); i++) {
    EXPECT_EQ(batch.item().Get(i).field().Get(0).token_id().size(),
        nTokens);
  }

  // Index doc-token matrix
  if (is_network_mode) artm::SaveBatch(batch, "00b6d631-46a6-4edf-8ef6-016c7b27d9f0.batch");

  std::shared_ptr<artm::TopicModel> topic_model;
  double expected_normalizer = 0;
  double previous_perplexity = 0;
  for (int iter = 0; iter < 5; ++iter) {
    if (is_network_mode) master_component->InvokeIteration();
    else master_component->AddBatch(batch, /*reset_scores =*/ true);  // NOLINT
    master_component->WaitIdle();
    model.Synchronize(0.0);

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
    std::shared_ptr< ::artm::PerplexityScore> perplexity =
      master_component->GetScoreAs< ::artm::PerplexityScore>(model, "PerplexityScore");

    if (!is_network_mode) {
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
    }

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
      if (!is_network_mode) {
        // Verify that normalizer does not grow starting from second iteration.
        // This confirms that the Instance::ForceResetScores() function works as expected.
        EXPECT_EQ(perplexity->normalizer(), expected_normalizer);
      }
    }
  }

  if (is_network_mode) master_component->InvokeIteration();
  else master_component->AddBatch(batch, /*reset_scores =*/ true);  // NOLINT

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
    if (!is_network_mode) {
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
    }

    args.clear_topic_name();
    args.mutable_batch()->CopyFrom(batch);
    if (!is_network_mode) {
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
  }

  model_config.set_name("model2_name");
  artm::Model model2(*master_component, model_config);
  if (!is_network_mode) {
    // Test overwrite topic model
    artm::TopicModel new_topic_model;
    new_topic_model.set_name(model2.name());
    new_topic_model.set_topics_count(nTopics);
    new_topic_model.add_token("my overwritten token");
    new_topic_model.add_token("my overwritten token2");
    auto weights = new_topic_model.add_token_weights();
    auto weights2 = new_topic_model.add_token_weights();
    for (int i = 0; i < nTopics; ++i) {
      weights->add_value(static_cast<float>(i));
      weights2->add_value(static_cast<float>(nTopics - i));
    }

    model2.Overwrite(new_topic_model);

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
  model_config2.add_topic_name(model_config.topic_name(1));
  model_config2.add_topic_name(model_config.topic_name(2));
  model_config2.add_topic_name(model_config.topic_name(3));
  model_config2.add_topic_name(model_config.topic_name(4));
  model_config2.set_name("model5_name");
  model3.Reconfigure(model_config2);

  model3.Synchronize(0.0);
  args.Clear();
  args.set_model_name("model5_name");
  auto new_topic_model4 = master_component->GetTopicModel(args);
  ASSERT_EQ(new_topic_model4->topics_count(), 4);
  ASSERT_EQ(new_topic_model4->topic_name_size(), 4);
  EXPECT_EQ(new_topic_model4->topic_name(0), model_config.topic_name(1));
  EXPECT_EQ(new_topic_model4->topic_name(1), model_config.topic_name(2));
  EXPECT_EQ(new_topic_model4->topic_name(2), model_config.topic_name(3));
  EXPECT_EQ(new_topic_model4->topic_name(3), model_config.topic_name(4));
}

// artm_tests.exe --gtest_filter=CppInterface.BasicTest_StandaloneMode
TEST(CppInterface, BasicTest_StandaloneMode) {
  BasicTest(false, false);
}

// artm_tests.exe --gtest_filter=CppInterface.BasicTest_StandaloneProxyMode
TEST(CppInterface, BasicTest_StandaloneProxyMode) {
  BasicTest(false, true);
}

// artm_tests.exe --gtest_filter=CppInterface.BasicTest_NetworkMode
TEST(CppInterface, BasicTest_NetworkMode) {
  BasicTest(true, false);
}

// artm_tests.exe --gtest_filter=CppInterface.BasicTest_NetworkProxyMode
TEST(CppInterface, BasicTest_NetworkProxyMode) {
  BasicTest(true, true);
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

// artm_tests.exe --gtest_filter=CppInterface.ProxyExceptions
TEST(CppInterface, ProxyExceptions) {
  artm::MasterComponentConfig master_config;
  artm::MasterProxyConfig master_proxy_config;
  master_proxy_config.set_node_connect_endpoint("tcp://localhost:5557");
  master_proxy_config.mutable_config()->CopyFrom(master_config);
  master_proxy_config.set_communication_timeout(10);

  ASSERT_THROW(artm::MasterComponent master_component(master_proxy_config),
    artm::NetworkException);
}

// artm_tests.exe --gtest_filter=CppInterface.WaitIdleTimeout
TEST(CppInterface, WaitIdleTimeout) {
  ::artm::MasterComponentConfig master_config;
  master_config.set_processor_queue_max_size(10000);
  ::artm::MasterComponent master(master_config);
  ::artm::ModelConfig model_config;
  model_config.set_name("model_config1");
  ::artm::Model model(master, model_config);
  ::artm::Batch batch;
  batch.set_id("00b6d631-46a6-4edf-8ef6-016c7b27d9f0");
  for (int i = 0; i < 1000; ++i)
    master.AddBatch(batch);
  EXPECT_FALSE(master.WaitIdle(0));
}
