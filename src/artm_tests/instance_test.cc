// Copyright 2014, Additive Regularization of Topic Models.

#include "boost/lexical_cast.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/uuid/uuid.hpp"
#include "boost/uuid/uuid_generators.hpp"
#include "boost/uuid/uuid_io.hpp"

#include "gtest/gtest.h"

#include "artm/messages.pb.h"
#include "artm/core/instance.h"
#include "artm/core/merger.h"
#include "artm/core/data_loader.h"
#include "artm/core/protobuf_helpers.h"

class InstanceTest : boost::noncopyable {
 public:
  std::shared_ptr<artm::core::Instance> instance() { return instance_; }

  InstanceTest() : instance_(nullptr) {
    instance_.reset(new ::artm::core::Instance(
      ::artm::MasterComponentConfig(), ::artm::core::MasterInstanceLocal));
  }

  ~InstanceTest() {}

  // Some way of generating a junk content..
  // If you call this, then you really shouldn't care which content it will be;
  // the only promise of this function is to generate a batch that will have fixed
  // number of items (nItems). Under normal parameters it will
  // also have nTokens of unique tokens, and each item won't exceed maxLength.
  std::shared_ptr<artm::Batch> GenerateBatch(int n_tokens, int n_items, int start_id,
                                             int max_length, int max_occurences) {
    std::shared_ptr<artm::Batch> batch(std::make_shared<artm::Batch>());
    batch->set_id("2f0d43c9-234f-4f8a-acdb-efe2f2c95d50");
    for (int i = 0; i < n_tokens; ++i) {
      std::stringstream str;
      str << "token" << i;
      batch->add_token(str.str());
    }

    int iToken = 0;
    int iLength = 0;
    int iOccurences = 0;

    for (int iItem = 0; iItem < n_items; ++iItem) {
      artm::Item* item = batch->add_item();
      item->set_id(start_id++);
      artm::Field* field = item->add_field();
      for (int i = 0; i <= iLength; ++i) {
        field->add_token_id(iToken);
        field->add_token_count(iOccurences + 1);

        iOccurences = (iOccurences + 1) % max_occurences;
        iToken = (iToken + 1) % n_tokens;
      }

      iLength = (iLength + 1) % max_length;
    }

    return batch;
  }

 private:
  std::shared_ptr<artm::core::Instance> instance_;
};

// artm_tests.exe --gtest_filter=Instance.*
TEST(Instance, Basic) {
  auto instance = std::make_shared< ::artm::core::Instance>(
    ::artm::MasterComponentConfig(), ::artm::core::MasterInstanceLocal);

  artm::Batch batch1;
  batch1.set_id("c722e9bd-28f8-4af0-a4fe-790681982a87");
  batch1.add_token("first token");
  batch1.add_token("second");
  for (int i = 0; i < 2; ++i) {
    artm::Item* item = batch1.add_item();
    artm::Field* field = item->add_field();
    field->add_token_id(i);
    field->add_token_count(i+1);
  }

  artm::AddBatchArgs args1;
  args1.mutable_batch()->CopyFrom(batch1);  // +1

  artm::Batch batch4;
  batch4.set_id("b60b35ba-50ba-4c88-9cb0-5aa83fd84297");
  batch4.add_token("second");
  batch4.add_token("last");
  artm::Item* item = batch4.add_item();
  artm::Field* field = item->add_field();
  for (int iToken = 0; iToken < 2; ++iToken) {
    field->add_token_id(iToken);
    field->add_token_count(iToken + 2);
  }

  artm::AddBatchArgs args4;
  args4.mutable_batch()->CopyFrom(batch4);  // +4

  artm::ModelConfig config;
  config.set_enabled(true);
  config.set_topics_count(3);
  artm::core::ModelName model_name =
    boost::lexical_cast<std::string>(boost::uuids::random_generator()());
  config.set_name(boost::lexical_cast<std::string>(model_name));
  instance->CreateOrReconfigureModel(config);

  for (int i = 0; i < 20; ++i) {
    instance->local_data_loader()->AddBatch(args1);
    instance->local_data_loader()->AddBatch(args4);

    ::artm::WaitIdleArgs wait_args;
    instance->local_data_loader()->WaitIdle(wait_args);
    ::artm::SynchronizeModelArgs sync_model_args;
    sync_model_args.set_model_name(model_name);
    sync_model_args.set_decay_weight(1.0);
    instance->merger()->ForceSynchronizeModel(sync_model_args);
  }

  config.set_enabled(false);
  for (int i = 0; i < 3; ++i) {
    config.add_topic_name("@topic_" + std::to_string(i));
  }
  config.set_topics_count(0);
  instance->CreateOrReconfigureModel(config);

  artm::TopicModel topic_model;
  artm::GetTopicModelArgs args;
  args.set_model_name(model_name);
  for (int i = 0; i < 3; ++i) {
    args.add_topic_name("@topic_" + std::to_string(i));
  }

  instance->merger()->RetrieveExternalTopicModel(args, &topic_model);
  EXPECT_EQ(topic_model.token_size(), 3);
  EXPECT_EQ(topic_model.topics_count(), 3);
  EXPECT_TRUE(artm::core::model_has_token(topic_model, artm::core::Token(artm::core::DefaultClass, "first token")));
  EXPECT_TRUE(artm::core::model_has_token(topic_model, artm::core::Token(artm::core::DefaultClass, "second")));
  EXPECT_TRUE(artm::core::model_has_token(topic_model, artm::core::Token(artm::core::DefaultClass, "last")));
  EXPECT_FALSE(artm::core::model_has_token(topic_model, artm::core::Token(artm::core::DefaultClass, "of cource!")));
}

// artm_tests.exe --gtest_filter=Instance.MultipleStreamsAndModels
TEST(Instance, MultipleStreamsAndModels) {
  InstanceTest test;

  // This setting will ensure that
  // - first model have  Token0, Token2, Token4,
  // - second model have Token1, Token3, Token6,
  auto batch = test.GenerateBatch(6, 6, 0, 1, 1);
  artm::AddBatchArgs add_args;
  add_args.mutable_batch()->CopyFrom(*batch);
  add_args.set_reset_scores(true);

  ::artm::MasterComponentConfig config;
  artm::Stream* s1 = config.add_stream();
  s1->set_type(artm::Stream_Type_ItemIdModulus);
  s1->set_modulus(2);
  s1->add_residuals(0);
  s1->set_name("train");
  artm::Stream* s2 = config.add_stream();
  s2->set_type(artm::Stream_Type_ItemIdModulus);
  s2->set_modulus(2);
  s2->add_residuals(1);
  s2->set_name("test");

  // In the little synthetic dataset created below
  // tokens in 'train' and 'test' sample won't overlap.
  // If we chose to calc perplexity on test sample
  // it will be zero, because none of test-sample tokens
  // are present in token-topic-matrix. Therefore,
  // using train sample to get non-zero perplexity score.
  ::artm::ScoreConfig score_config;
  ::artm::PerplexityScoreConfig perplexity_config;
  perplexity_config.set_stream_name("train");
  score_config.set_config(perplexity_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_Perplexity);
  score_config.set_name("perplexity");
  config.add_score_config()->CopyFrom(score_config);

  test.instance()->Reconfigure(config);

  artm::ModelConfig m1;
  m1.set_stream_name("train");
  m1.set_enabled(true);
  m1.set_name(boost::lexical_cast<std::string>(boost::uuids::random_generator()()));
  m1.add_score_name("perplexity");
  test.instance()->CreateOrReconfigureModel(m1);

  artm::ModelConfig m2;
  m2.set_stream_name("test");
  m2.set_enabled(true);
  m2.set_name(boost::lexical_cast<std::string>(boost::uuids::random_generator()()));
  test.instance()->CreateOrReconfigureModel(m2);

  for (int iter = 0; iter < 5; ++iter) {
    test.instance()->local_data_loader()->AddBatch(add_args);
    test.instance()->local_data_loader()->WaitIdle(artm::WaitIdleArgs());
    test.instance()->merger()->ForceSynchronizeModel(::artm::SynchronizeModelArgs());
  }

  artm::TopicModel m1t;
  artm::GetTopicModelArgs args;
  args.set_model_name(m1.name());
  test.instance()->merger()->RetrieveExternalTopicModel(args, &m1t);

  artm::TopicModel m2t;
  args.set_model_name(m2.name());
  test.instance()->merger()->RetrieveExternalTopicModel(args, &m2t);

  artm::ScoreData m1score_data;
  artm::GetScoreValueArgs get_score_args;
  get_score_args.set_model_name(m1.name());
  get_score_args.set_score_name("perplexity");
  test.instance()->merger()->RequestScore(get_score_args, &m1score_data);
  artm::PerplexityScore perplexity_score;
  perplexity_score.ParseFromString(m1score_data.data());

  // Verification for m1t (the first model)
  EXPECT_TRUE(artm::core::model_has_token(m1t, artm::core::Token(artm::core::DefaultClass, "token0")));
  EXPECT_TRUE(artm::core::model_has_token(m1t, artm::core::Token(artm::core::DefaultClass, "token2")));
  EXPECT_TRUE(artm::core::model_has_token(m1t, artm::core::Token(artm::core::DefaultClass, "token4")));

  // if model has other tokens, their Phi weight should be at zero.
  for (int token_index = 0; token_index < m1t.token_size(); ++token_index) {
    std::string token = m1t.token(token_index);
    if ((token == "token1") || (token == "token3") || (token == "token5")) {
      for (int topic_index = 0; topic_index < m1t.topics_count(); ++topic_index) {
        // todo(alfrey) Verification was disabled because now all tokens are initialized with random values.
        // EXPECT_EQ(m1t.token_weights(token_index).value(topic_index), 0);
      }
    }
  }

  // Verification for m2t (the second model)
  EXPECT_TRUE(artm::core::model_has_token(m1t, artm::core::Token(artm::core::DefaultClass, "token1")));
  EXPECT_TRUE(artm::core::model_has_token(m1t, artm::core::Token(artm::core::DefaultClass, "token3")));
  EXPECT_TRUE(artm::core::model_has_token(m1t, artm::core::Token(artm::core::DefaultClass, "token5")));

  // if model has other tokens, their Phi weight should be at zero.
  for (int token_index = 0; token_index < m2t.token_size(); ++token_index) {
    std::string token = m2t.token(token_index);
    if ((token == "token0") || (token == "token2") || (token == "token4")) {
      for (int topic_index = 0; topic_index < m2t.topics_count(); ++topic_index) {
        // todo(alfrey) Verification was disabled because now all tokens are initialized with random values.
        // EXPECT_EQ(m2t.token_weights(token_index).value(topic_index), 0);
      }
    }
  }

  EXPECT_GT(perplexity_score.value(), 0);
}
