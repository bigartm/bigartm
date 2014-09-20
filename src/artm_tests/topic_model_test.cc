// Copyright 2014, Additive Regularization of Topic Models.

#include "gtest/gtest.h"

#include "artm/core/common.h"
#include "artm/core/topic_model.h"
#include "artm/messages.pb.h"

TEST(TopicModelTest, Basic) {
  const float kTolerance = 1e-5f;

  int no_topics = 3;
  int no_tokens = 5;

  artm::core::TopicModel topic_model(::artm::core::ModelName(), no_topics);
  topic_model.AddToken(artm::core::Token(artm::core::DefaultClass, "token_1"));
  topic_model.AddToken(artm::core::Token(artm::core::DefaultClass, "token_2"));
  topic_model.AddToken(artm::core::Token(artm::core::DefaultClass, "token_3"));
  topic_model.AddToken(artm::core::Token(artm::core::DefaultClass, "token_4"));
  topic_model.AddToken(artm::core::Token(artm::core::DefaultClass, "token_5"));

  //  test 1
  float real_normalizer = 0;
  for (int i = 0; i < no_tokens; ++i) {
    for (int j = 0; j < no_topics; ++j) {
      topic_model.SetTokenWeight(i, j, 1);
    }
  }
  auto n_t = topic_model.GetTopicWeightIterator(0).GetNormalizer();
  for (int j = 0; j < no_topics; ++j) {
    real_normalizer += n_t[j];
  }
  float expected_normalizer = static_cast<float>(no_tokens * no_topics);
  EXPECT_TRUE(std::abs(real_normalizer - expected_normalizer) < kTolerance);

  //  test 2
  real_normalizer = 0;
  for (int i = 0; i < no_tokens; ++i) {
    for (int j = 0; j < no_topics; ++j) {
      topic_model.SetRegularizerWeight(i, j, -0.5f);
    }
  }
  n_t = topic_model.GetTopicWeightIterator(0).GetNormalizer();
  for (int j = 0; j < no_topics; ++j) {
    real_normalizer += n_t[j];
  }
  expected_normalizer = no_tokens * no_topics / 2.0f;
  EXPECT_TRUE(std::abs(real_normalizer - expected_normalizer) < kTolerance);

  //  test 3
  real_normalizer = 0;
  for (int i = 0; i < no_tokens; ++i) {
    for (int j = 0; j < no_topics; ++j) {
      topic_model.SetRegularizerWeight(i, j, -1.5);
    }
  }
  n_t = topic_model.GetTopicWeightIterator(0).GetNormalizer();
  for (int j = 0; j < no_topics; ++j) {
    real_normalizer += n_t[j];
  }
  expected_normalizer = 0;
  EXPECT_TRUE(std::abs(real_normalizer - expected_normalizer) < kTolerance);

  //  test 4
  real_normalizer = 0;
  for (int i = 0; i < no_tokens; ++i) {
    for (int j = 0; j < no_topics; ++j) {
      topic_model.IncreaseTokenWeight(i, j, 0.4f);
    }
  }
  n_t = topic_model.GetTopicWeightIterator(0).GetNormalizer();
  for (int j = 0; j < no_topics; ++j) {
    real_normalizer += n_t[j];
  }
  expected_normalizer = 0;
  EXPECT_TRUE(std::abs(real_normalizer - expected_normalizer) < kTolerance);

  //  test 5
  real_normalizer = 0;
  for (int i = 0; i < no_tokens; ++i) {
    for (int j = 0; j < no_topics; ++j) {
      topic_model.IncreaseTokenWeight(i, j, 0.6f);
    }
  }
  n_t = topic_model.GetTopicWeightIterator(0).GetNormalizer();
  for (int j = 0; j < no_topics; ++j) {
    real_normalizer += n_t[j];
  }
  expected_normalizer = no_tokens * no_topics / 2.0f;
  EXPECT_TRUE(std::abs(real_normalizer - expected_normalizer) < kTolerance);

  //  test 6
  real_normalizer = 0;
  for (int i = 0; i < no_tokens; ++i) {
    for (int j = 0; j < no_topics; ++j) {
      topic_model.SetTokenWeight(i, j, 1);
    }
  }
  n_t = topic_model.GetTopicWeightIterator(0).GetNormalizer();
  for (int j = 0; j < no_topics; ++j) {
    real_normalizer += n_t[j];
  }
  expected_normalizer = 0;
  EXPECT_TRUE(std::abs(real_normalizer - expected_normalizer) < kTolerance);

  //  test 7
  real_normalizer = 0;
  for (int i = 0; i < no_tokens; ++i) {
    for (int j = 0; j < no_topics; ++j) {
      topic_model.SetRegularizerWeight(i, j, -0.5f);
    }
  }
  n_t = topic_model.GetTopicWeightIterator(0).GetNormalizer();
  for (int j = 0; j < no_topics; ++j) {
    real_normalizer += n_t[j];
  }
  expected_normalizer = no_tokens * no_topics / 2.0f;
  EXPECT_TRUE(std::abs(real_normalizer - expected_normalizer) < kTolerance);

  //  test 8
  no_topics = 1;
  for (int i = 1; i < 10; ++i) {
    artm::core::TopicModel topic_model_1(::artm::core::ModelName(), no_topics);
    topic_model_1.AddToken(artm::core::Token(artm::core::DefaultClass, "token_1"));
    topic_model_1.AddToken(artm::core::Token(artm::core::DefaultClass, "token_2"));
    topic_model_1.AddToken(artm::core::Token(artm::core::DefaultClass, "token_3"));
    topic_model_1.AddToken(artm::core::Token(artm::core::DefaultClass, "token_4"));
    topic_model_1.AddToken(artm::core::Token(artm::core::DefaultClass, "token_5"));

    for (int j = 0; j < 100; ++j) {
      int index = 0 + rand() % 5;  // NOLINT
      int func = 0 + rand() % 4;   // NOLINT
      float value = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2));  // NOLINT
      switch (func) {
      case 0:
        topic_model_1.SetRegularizerWeight(index, 0, value);
        break;
      case 1:
        topic_model_1.SetTokenWeight(index, 0, value);
        break;
      case 2:
        topic_model_1.IncreaseTokenWeight(index, 0, value);
        break;
      case 4:
        topic_model_1.IncreaseRegularizerWeight(index, 0, value);
        break;
      }

      float expected_norm = 0;
      float real_norm = 0;
      for (int token_id = 0; token_id < no_tokens; ++token_id) {
        auto iter = topic_model_1.GetTopicWeightIterator(token_id);
        iter.NextTopic();
        float r = (iter.GetRegularizer())[0];
        float n = (iter.GetData())[0];
        expected_norm = (iter.GetNormalizer())[0];
        if (r + n > 0.0) {
          real_norm += (r + n);
        }
      }
      EXPECT_TRUE(std::abs(real_norm - expected_norm) < kTolerance);
    }
  }
}
