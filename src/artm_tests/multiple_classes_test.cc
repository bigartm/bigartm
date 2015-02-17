// Copyright 2014, Additive Regularization of Topic Models.

#include "boost/thread.hpp"
#include "gtest/gtest.h"

#include "boost/filesystem.hpp"

#include "artm/cpp_interface.h"
#include "artm/core/exceptions.h"
#include "artm/messages.pb.h"

#include "artm/core/internals.pb.h"

void ShowTopicModel(const ::artm::TopicModel& topic_model) {
  for (int i = 0; i < topic_model.token_size(); ++i) {
    if (i > 10) break;
    std::cout << topic_model.token(i) << "(" << topic_model.class_id(i) << "): ";
    const ::artm::FloatArray& weights = topic_model.token_weights(i);
    for (int j = 0; j < weights.value_size(); ++j)
      std::cout << std::fixed << std::setw(4) << std::setprecision(3) << weights.value(j) << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void ShowThetaMatrix(const ::artm::ThetaMatrix& theta_matrix) {
  for (int i = 0; i < theta_matrix.item_id_size(); ++i) {
    if (i > 10) break;
    std::cout << theta_matrix.item_id(i) << ": ";
    const artm::FloatArray& weights = theta_matrix.item_weights(i);
    for (int j = 0; j < weights.value_size(); ++j) {
      std::cout << std::fixed << std::setw(4) << std::setprecision(3) << weights.value(j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

bool CompareTopicModels(const ::artm::TopicModel& t1, const ::artm::TopicModel& t2, float* max_diff) {
  *max_diff = 0.0f;
  if (t1.token_size() != t2.token_size()) return false;
  for (int i = 0; i < t1.token_size(); ++i) {
    if (t1.token(i) != t2.token(i)) return false;
    if (t1.class_id(i) != t2.class_id(i)) return false;
    const artm::FloatArray& w1 = t1.token_weights(i);
    const artm::FloatArray& w2 = t2.token_weights(i);
    if (w1.value_size() != w2.value_size()) return false;
    for (int j = 0; j < w1.value_size(); ++j) {
      float diff = fabs(w1.value(j) - w2.value(j));
      if (diff > *max_diff) *max_diff = diff;
    }
  }

  return true;
}

bool CompareThetaMatrices(const ::artm::ThetaMatrix& t1, const ::artm::ThetaMatrix& t2, float *max_diff) {
  *max_diff = 0.0f;
  if (t1.item_id_size() != t2.item_id_size()) return false;
  for (int i = 0; i < t1.item_id_size(); ++i) {
    if (t1.item_id(i) != t2.item_id(i)) return false;
    const artm::FloatArray& w1 = t1.item_weights(i);
    const artm::FloatArray& w2 = t2.item_weights(i);
    if (w1.value_size() != w2.value_size()) return false;
    for (int j = 0; j < w1.value_size(); ++j) {
      float diff = (fabs(w1.value(j) - w2.value(j)));
      if (diff > *max_diff) *max_diff = diff;
    }
  }

  return true;
}

artm::Batch GenerateBatch(int nTokens, int nDocs, std::string class1, std::string class2) {
  artm::Batch batch;
  batch.set_id("11972762-6a23-4524-b089-7122816aff72");
  for (int i = 0; i < nTokens; i++) {
    std::stringstream str;
    str << "token" << i;
    std::string class_id = (i % 2 == 0) ? class1 : class2;
    batch.add_token(str.str());
    batch.add_class_id(class_id);
  }

  for (int iDoc = 0; iDoc < nDocs; iDoc++) {
    artm::Item* item = batch.add_item();
    item->set_id(iDoc);
    artm::Field* field = item->add_field();
    for (int iToken = 0; iToken < nTokens; ++iToken) {
      field->add_token_id(iToken);
      int background_count = (iToken > 40) ? (1 + rand() % 5) : 0;  // NOLINT
      int topical_count = ((iToken < 40) && ((iToken % 10) == (iDoc % 10))) ? 10 : 0;
      field->add_token_count(background_count + topical_count);
    }
  }

  return batch;
}

// artm_tests.exe --gtest_filter=MultipleClasses.BasicTest
TEST(MultipleClasses, BasicTest) {
  ::artm::MasterComponentConfig master_config;
  master_config.set_cache_theta(true);
  ::artm::MasterComponent master_component(master_config);

  // Create theta-regularizer for some (not all) topics
  ::artm::RegularizerConfig regularizer_config;
  regularizer_config.set_name("regularizer_smsp_theta");
  regularizer_config.set_type(::artm::RegularizerConfig_Type_SmoothSparseTheta);
  ::artm::SmoothSparseThetaConfig smooth_sparse_theta_config;
  smooth_sparse_theta_config.add_topic_name("Topic3");
  smooth_sparse_theta_config.add_topic_name("Topic7");
  regularizer_config.set_config(smooth_sparse_theta_config.SerializeAsString());
  ::artm::Regularizer regularizer_smsp_theta(master_component, regularizer_config);

  // Generate doc-token matrix
  int nTokens = 60;
  int nDocs = 100;
  int nTopics = 10;

  artm::Batch batch = GenerateBatch(nTokens, nDocs, "@default_class", "__custom_class");
  artm::TopicModel initial_model;
  for (int i = 0; i < batch.token_size(); i++) {
    initial_model.add_token(batch.token(i));
    initial_model.add_class_id(batch.class_id(i));
    initial_model.set_topics_count(nTopics);
    ::artm::FloatArray* token_weights = initial_model.add_token_weights();
    for (int topic_index = 0; topic_index < nTopics; ++topic_index) {
      token_weights->add_value(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));  // NOLINT
    }
  }

  // Create model
  artm::ModelConfig model_config1, model_config2, model_config3;
  model_config1.set_name("model1"); model_config1.set_topics_count(nTopics);
  model_config2.set_name("model2"); model_config2.set_topics_count(nTopics); model_config2.set_use_sparse_bow(false);
  model_config3.set_name("model3"); model_config3.set_topics_count(nTopics);
  model_config3.add_class_id("@default_class"); model_config3.add_class_weight(0.5f);
  model_config3.add_class_id("__custom_class"); model_config3.add_class_weight(2.0f);

  artm::Model model1(master_component, model_config1);
  artm::Model model2(master_component, model_config2);
  artm::Model model3(master_component, model_config3);
  model1.Overwrite(initial_model); model2.Overwrite(initial_model); model3.Overwrite(initial_model);

  // Create a regularized model
  artm::ModelConfig model_config_reg;
  model_config_reg.set_name("model_config_reg");
  model_config_reg.add_regularizer_name("regularizer_smsp_theta");
  model_config_reg.add_regularizer_tau(-1.0);
  for (int i = 0; i < nTopics; ++i) {
    std::stringstream ss;
    ss << "Topic" << i;
    model_config_reg.add_topic_name(ss.str());
  }
  artm::Model model_reg(master_component, model_config_reg);
  model_reg.Overwrite(initial_model);

  // Index doc-token matrix
  int nIters = 5;
  std::shared_ptr< ::artm::ThetaMatrix> theta_matrix1_explicit, theta_matrix2_explicit, theta_matrix3_explicit;
  for (int iter = 0; iter < 5; ++iter) {
    if (iter == (nIters - 1)) {
      // Now we would like to verify that master_component.GetThetaMatrix gives the same result in two cases:
      // 1. Retrieving ThetaMatrix cached on the last iteration  (done in Processor::ThreadFunction() method)
      // 2. Explicitly getting ThateMatrix for the batch (done in Processor::FindThetaMatrix() method)
      // These results should be identical only if the same version of TopicModel is used in both cases.
      // This imply that we should cached theta matrix with GetThetaMatrix(batch) at the one-before-last iteration.
      // An alternative would be to not invoke model.Synchronize on the last iteration.
      ::artm::GetThetaMatrixArgs gtm1, gtm2, gtm3;
      gtm1.mutable_batch()->CopyFrom(batch); gtm1.set_model_name(model1.name());
      gtm2.mutable_batch()->CopyFrom(batch); gtm2.set_model_name(model2.name());
      gtm3.mutable_batch()->CopyFrom(batch); gtm3.set_model_name(model3.name());
      theta_matrix1_explicit = master_component.GetThetaMatrix(gtm1);
      theta_matrix2_explicit = master_component.GetThetaMatrix(gtm2);
      theta_matrix3_explicit = master_component.GetThetaMatrix(gtm3);
    }
    master_component.AddBatch(batch, true);
    master_component.WaitIdle();
    model1.Synchronize(0.0);
    model2.Synchronize(0.0);
    model3.Synchronize(0.0);
    model_reg.Synchronize(0.0);
  }

  std::shared_ptr< ::artm::TopicModel> topic_model1 = master_component.GetTopicModel(model1.name());
  std::shared_ptr< ::artm::TopicModel> topic_model2 = master_component.GetTopicModel(model2.name());
  std::shared_ptr< ::artm::TopicModel> topic_model3 = master_component.GetTopicModel(model3.name());
  std::shared_ptr< ::artm::TopicModel> topic_model_reg = master_component.GetTopicModel(model_reg.name());

  std::shared_ptr< ::artm::ThetaMatrix> theta_matrix1 = master_component.GetThetaMatrix(model1.name());
  std::shared_ptr< ::artm::ThetaMatrix> theta_matrix2 = master_component.GetThetaMatrix(model2.name());
  std::shared_ptr< ::artm::ThetaMatrix> theta_matrix3 = master_component.GetThetaMatrix(model3.name());
  std::shared_ptr< ::artm::ThetaMatrix> theta_matrix_reg = master_component.GetThetaMatrix(model_reg.name());

  ShowTopicModel(*topic_model1);
  ShowTopicModel(*topic_model2);
  ShowTopicModel(*topic_model3);
  ShowTopicModel(*topic_model_reg);

  // ShowThetaMatrix(*theta_matrix1);
  // ShowThetaMatrix(*theta_matrix1_explicit);
  // ShowThetaMatrix(*theta_matrix2);
  // ShowThetaMatrix(*theta_matrix2_explicit);
  // ShowThetaMatrix(*theta_matrix3);
  // ShowThetaMatrix(*theta_matrix3_explicit);
  // ShowThetaMatrix(*theta_matrix_reg);  // <- 3 and 7 topics should be sparse in this matrix.

  float max_diff;
  // Compare consistency between Theta calculation in Processor::ThreadFunction() and Processor::FindThetaMatrix()
  EXPECT_TRUE(CompareThetaMatrices(*theta_matrix1, *theta_matrix1_explicit, &max_diff));
  EXPECT_LT(max_diff, 0.001);  // "theta_matrix1 == theta_matrix1_explicit");
  EXPECT_TRUE(CompareThetaMatrices(*theta_matrix2, *theta_matrix2_explicit, &max_diff));
  EXPECT_LT(max_diff, 0.001);  // "theta_matrix2 == theta_matrix2_explicit");
  EXPECT_TRUE(CompareThetaMatrices(*theta_matrix3, *theta_matrix3_explicit, &max_diff));
  EXPECT_LT(max_diff, 0.001);  // "theta_matrix3 == theta_matrix3_explicit");

  // Compare consistency between use_sparse_bow==true and use_sparse_bow==false
  EXPECT_TRUE(CompareTopicModels(*topic_model1, *topic_model2, &max_diff));
  //  EXPECT_LT(max_diff, 0.001);  // topic_model1 == topic_model2
  EXPECT_TRUE(CompareThetaMatrices(*theta_matrix1, *theta_matrix2, &max_diff));
  //  EXPECT_LT(max_diff, 0.001);  // "theta_matrix1 == theta_matrix2");

  // Verify that changing class_weight has an effect on the resulting model
  EXPECT_TRUE(CompareTopicModels(*topic_model3, *topic_model1, &max_diff));
  EXPECT_GT(max_diff, 0.001);  // topic_model3 != topic_model1

  EXPECT_TRUE(CompareThetaMatrices(*theta_matrix3, *theta_matrix1, &max_diff));
  EXPECT_GT(max_diff, 0.001);  // "theta_matrix3 != theta_matrix1");
}

void configureTopTokensScore(std::string score_name, std::string class_id, artm::MasterComponentConfig* master_config) {
  ::artm::ScoreConfig score_config;
  ::artm::TopTokensScoreConfig top_tokens_config;
  top_tokens_config.set_num_tokens(4);
  if (!class_id.empty()) top_tokens_config.set_class_id(class_id);
  score_config.set_config(top_tokens_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_TopTokens);
  score_config.set_name(score_name);
  master_config->add_score_config()->CopyFrom(score_config);
}

void configurePerplexityScore(std::string score_name, artm::MasterComponentConfig* master_config) {
  ::artm::ScoreConfig score_config;
  ::artm::PerplexityScoreConfig perplexity_config;
  score_config.set_config(perplexity_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_Perplexity);
  score_config.set_name(score_name);
  master_config->add_score_config()->CopyFrom(score_config);
}

void configureThetaSnippetScore(std::string score_name, int num_items, artm::MasterComponentConfig* master_config) {
  ::artm::ScoreConfig score_config;
  ::artm::ThetaSnippetScoreConfig theta_snippet_config;
  for (int i = 0; i < num_items; ++i)
    theta_snippet_config.add_item_id(i);
  score_config.set_config(theta_snippet_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_ThetaSnippet);
  score_config.set_name(score_name);
  master_config->add_score_config()->CopyFrom(score_config);
}

void configureItemsProcessedScore(std::string score_name, artm::MasterComponentConfig* master_config) {
  ::artm::ScoreConfig score_config;
  ::artm::ItemsProcessedScore items_processed_config;
  score_config.set_config(items_processed_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_ItemsProcessed);
  score_config.set_name(score_name);
  master_config->add_score_config()->CopyFrom(score_config);
}

void PrintTopTokenScore(const ::artm::TopTokensScore& top_tokens) {
  int topic_index = -1;
  for (int i = 0; i < top_tokens.num_entries(); i++) {
    if (top_tokens.topic_index(i) != topic_index) {
      topic_index = top_tokens.topic_index(i);
      std::cout << "\n#" << (topic_index + 1) << ": ";
    }

    std::cout << top_tokens.token(i) << "(" << std::setw(2) << std::setprecision(2) << top_tokens.weight(i) << ") ";
  }
}

// artm_tests.exe --gtest_filter=MultipleClasses.WithoutDefaultClass
TEST(MultipleClasses, WithoutDefaultClass) {
  ::artm::MasterComponentConfig master_config;
  configureTopTokensScore("default_class", "", &master_config);
  configureTopTokensScore("tts_class_one", "class_one", &master_config);
  configureTopTokensScore("tts_class_two", "class_two", &master_config);
  configureThetaSnippetScore("theta_snippet", /*num_items = */ 5, &master_config);
  configurePerplexityScore("perplexity", &master_config);
  configureItemsProcessedScore("items_processed", &master_config);
  ::artm::MasterComponent master_component(master_config);

  // Generate doc-token matrix
  int nTokens = 60, nDocs = 100, nTopics = 10;
  artm::Batch batch = GenerateBatch(nTokens, nDocs, "class_one", "class_two");

  artm::ModelConfig model_config1;
  model_config1.set_name("model1"); model_config1.set_topics_count(nTopics);
  model_config1.add_class_id("class_one"); model_config1.add_class_weight(2.0f);
  // model_config1.add_score_name("default_class"); model_config1.add_score_name("tts_class_one");
  // model_config1.add_score_name("tts_class_two");
  model_config1.add_score_name("perplexity");
  model_config1.add_score_name("theta_snippet");
  model_config1.add_score_name("items_processed");
  artm::Model model1(master_component, model_config1);

  artm::ModelConfig model_config2;
  model_config2.set_name("model2"); model_config2.set_topics_count(nTopics);
  model_config2.add_class_id("class_one"); model_config2.add_class_weight(2.0f);
  model_config2.add_class_id("class_two"); model_config2.add_class_weight(0.5f);
  // model_config2.add_score_name("default_class"); model_config2.add_score_name("tts_class_one");
  // model_config2.add_score_name("tts_class_two");
  model_config2.add_score_name("perplexity");
  artm::Model model2(master_component, model_config2);

  for (int iter = 0; iter < 5; ++iter) {
    master_component.AddBatch(batch, /*bool reset_scores=*/ true);
    master_component.WaitIdle();
    model1.Synchronize(0.0);
    model2.Synchronize(0.0);
  }

  std::shared_ptr< ::artm::TopicModel> topic_model1 = master_component.GetTopicModel(model1.name());
  std::shared_ptr< ::artm::TopicModel> topic_model2 = master_component.GetTopicModel(model2.name());
  EXPECT_EQ(topic_model1->token_size(), 30);
  EXPECT_EQ(topic_model2->token_size(), 60);
  // ShowTopicModel(*topic_model1);
  // ShowTopicModel(*topic_model2);

  EXPECT_EQ(master_component.GetScoreAs< ::artm::TopTokensScore>(model1, "default_class")->num_entries(), 0);
  EXPECT_TRUE(master_component.GetScoreAs< ::artm::TopTokensScore>(model1, "tts_class_one")->num_entries() > 0);
  EXPECT_EQ(master_component.GetScoreAs< ::artm::TopTokensScore>(model1, "tts_class_two")->num_entries(), 0);
  EXPECT_EQ(master_component.GetScoreAs< ::artm::TopTokensScore>(model2, "default_class")->num_entries(), 0);
  EXPECT_TRUE(master_component.GetScoreAs< ::artm::TopTokensScore>(model2, "tts_class_one")->num_entries() > 0);
  EXPECT_TRUE(master_component.GetScoreAs< ::artm::TopTokensScore>(model2, "tts_class_two")->num_entries() > 0);

  float p1 = master_component.GetScoreAs< ::artm::PerplexityScore>(model1, "perplexity")->value();
  float p2 = master_component.GetScoreAs< ::artm::PerplexityScore>(model2, "perplexity")->value();
  EXPECT_TRUE((p1 > 0) && (p2 > 0) && (p1 != p2));

  auto theta_snippet = master_component.GetScoreAs< ::artm::ThetaSnippetScore>(model1, "theta_snippet");
  EXPECT_EQ(theta_snippet->item_id_size(), 5);
  EXPECT_EQ(master_component.GetScoreAs< ::artm::ItemsProcessedScore>(model1, "items_processed")->value(), nDocs);
}
