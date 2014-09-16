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

  std::shared_ptr<::artm::NodeController> node_controller;
  std::shared_ptr<::artm::NodeController> node_controller_master;
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
  model_config.add_score_name("PerplexityScore");
  model_config.add_regularizer_name(reg_decor_name);
  model_config.add_regularizer_tau(1);
  model_config.add_regularizer_name(reg_multilang_name);
  model_config.add_regularizer_tau(1);
  artm::Model model(*master_component, model_config);

  // Load doc-token matrix
  int nTokens = 10;
  int nDocs = 15;

  artm::Batch batch;
  for (int i = 0; i < nTokens; i++) {
    std::stringstream str;
    str << "token" << i;
    batch.add_token(str.str());
  }

  for (int iDoc = 0; iDoc < nDocs; iDoc++) {
    artm::Item* item = batch.add_item();
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
  master_component->AddBatch(batch);

  std::shared_ptr<artm::TopicModel> topic_model;
  double expected_normalizer = 0;
  for (int iter = 0; iter < 5; ++iter) {
    master_component->InvokeIteration(1);
    master_component->WaitIdle();
    topic_model = master_component->GetTopicModel(model);
    std::shared_ptr<::artm::PerplexityScore> perplexity =
      master_component->GetScoreAs<::artm::PerplexityScore>(model, "PerplexityScore");

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

  master_component->InvokeIteration(1);
  EXPECT_TRUE(master_component->WaitIdle());

  auto old_state_wrapper = master_component->GetRegularizerState(reg_multilang_name);
  model.InvokePhiRegularizers();
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
  EXPECT_EQ(nUniqueTokens, topic_model->token_size());
  auto first_token_topics = topic_model->token_weights(0);
  EXPECT_EQ(first_token_topics.value_size(), nTopics);

  if (!is_network_mode) {
    std::shared_ptr<::artm::ThetaMatrix> theta_matrix = master_component->GetThetaMatrix(model);
    EXPECT_TRUE(theta_matrix->item_id_size() == nDocs);
    for (int item_index = 0; item_index < theta_matrix->item_id_size(); ++item_index) {
      const ::artm::FloatArray& weights = theta_matrix->item_weights(item_index);
      EXPECT_EQ(weights.value_size(), nTopics);
      float sum = 0;
      for (int topic_index = 0; topic_index < weights.value_size(); ++topic_index) {
        float weight = weights.value(topic_index);
        EXPECT_GT(weight, 0);
        sum += weight;
      }

      EXPECT_LE(abs(sum - 1), 0.001);
    }
  }

  if (!is_network_mode) {
    // Test overwrite topic model
    artm::TopicModel new_topic_model;
    new_topic_model.set_name(model.name());
    new_topic_model.set_topics_count(nTopics);
    new_topic_model.add_token("my overwritten token");
    new_topic_model.add_token("my overwritten token2");
    auto weights = new_topic_model.add_token_weights();
    auto weights2 = new_topic_model.add_token_weights();
    for (int i = 0; i < nTopics; ++i) {
      weights->add_value(static_cast<float>(i));
      weights2->add_value(static_cast<float>(nTopics - i));
    }

    model.Overwrite(new_topic_model);
    auto new_topic_model2 = master_component->GetTopicModel(model);
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
  artm::Model model(master_component, model_config);

  model.mutable_config()->set_topics_count(20);
  ASSERT_THROW(model.Reconfigure(model.config()), artm::InvalidOperationException);
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

TEST(CppInterface, WaitIdleTimeout) {
  ::artm::MasterComponentConfig master_config;
  ::artm::MasterComponent master(master_config);
  master.AddBatch(::artm::Batch());
  master.InvokeIteration(10000);
  EXPECT_FALSE(master.WaitIdle(1));
}
