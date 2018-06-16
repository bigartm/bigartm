// Copyright 2017, Additive Regularization of Topic Models.

#include <sstream>  // NOLINT

#include "boost/thread.hpp"
#include "gtest/gtest.h"

#include "boost/filesystem.hpp"

#include "artm/cpp_interface.h"
#include "artm/core/exceptions.h"
#include "artm/core/common.h"

#include "artm_tests/test_mother.h"
#include "artm_tests/api.h"

void ShowTopicModel(const ::artm::TopicModel& topic_model) {
  for (int i = 0; i < topic_model.token_size(); ++i) {
    if (i > 10) {
      break;
    }
    std::cout << topic_model.token(i) << "(" << topic_model.class_id(i) << "): ";
    const ::artm::FloatArray& weights = topic_model.token_weights(i);
    for (int j = 0; j < weights.value_size(); ++j) {
      std::cout << std::fixed << std::setw(4) << std::setprecision(3) << weights.value(j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void ShowThetaMatrix(const ::artm::ThetaMatrix& theta_matrix) {
  for (int i = 0; i < theta_matrix.item_id_size(); ++i) {
    if (i > 10) {
      break;
    }
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
  if (t1.token_size() != t2.token_size()) {
    return false;
  }

  for (int i = 0; i < t1.token_size(); ++i) {
    if (t1.token(i) != t2.token(i) || t1.class_id(i) != t2.class_id(i)) {
      return false;
    }

    const artm::FloatArray& w1 = t1.token_weights(i);
    const artm::FloatArray& w2 = t2.token_weights(i);

    if (w1.value_size() != w2.value_size()) {
      return false;
    }

    for (int j = 0; j < w1.value_size(); ++j) {
      float diff = fabs(w1.value(j) - w2.value(j));
      if (diff > *max_diff) {
        *max_diff = diff;
      }
    }
  }

  return true;
}

bool CompareThetaMatrices(const ::artm::ThetaMatrix& t1, const ::artm::ThetaMatrix& t2, float *max_diff) {
  *max_diff = 0.0f;
  if (t1.item_id_size() != t2.item_id_size()) {
    return false;
  }

  for (int i = 0; i < t1.item_id_size(); ++i) {
    if (t1.item_id(i) != t2.item_id(i)) {
      return false;
    }

    const artm::FloatArray& w1 = t1.item_weights(i);
    const artm::FloatArray& w2 = t2.item_weights(i);

    if (w1.value_size() != w2.value_size()) {
      return false;
    }

    for (int j = 0; j < w1.value_size(); ++j) {
      float diff = (fabs(w1.value(j) - w2.value(j)));
      if (diff > *max_diff) {
        *max_diff = diff;
      }
    }
  }

  return true;
}

// artm_tests.exe --gtest_filter=MultipleClasses.BasicTest
TEST(MultipleClasses, BasicTest) {
  int nTokens = 60;
  int nDocs = 100;
  int nTopics = 10;

  ::artm::MasterModelConfig master_config = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  master_config.set_cache_theta(true);

  ::artm::MasterModelConfig master_config3(master_config);
  master_config3.add_class_id("@default_class"); master_config3.add_class_weight(0.5f);
  master_config3.add_class_id("__custom_class"); master_config3.add_class_weight(2.0f);

  ::artm::MasterModelConfig master_config_reg(master_config);
  // Create theta-regularizer for some (not all) topics
  ::artm::RegularizerConfig* regularizer_config = master_config_reg.add_regularizer_config();
  regularizer_config->set_name("regularizer_smsp_theta");
  regularizer_config->set_type(::artm::RegularizerType_SmoothSparseTheta);
  ::artm::SmoothSparseThetaConfig smooth_sparse_theta_config;
  smooth_sparse_theta_config.add_topic_name("Topic3");
  smooth_sparse_theta_config.add_topic_name("Topic7");
  regularizer_config->set_config(smooth_sparse_theta_config.SerializeAsString());
  regularizer_config->set_tau(-1.0);

  // Create ptdw-regularizer
  ::artm::RegularizerConfig* regularizer_config2 = master_config_reg.add_regularizer_config();
  regularizer_config2->set_name("regularizer_ptdw");
  regularizer_config2->set_type(::artm::RegularizerType_SmoothPtdw);
  ::artm::SmoothPtdwConfig smooth_ptdw_config;
  smooth_ptdw_config.set_window(5);
  regularizer_config2->set_config(smooth_ptdw_config.SerializeAsString());
  regularizer_config2->set_tau(2.0);

  // Generate doc-token matrix
  artm::Batch batch = ::artm::test::Helpers::GenerateBatch(nTokens, nDocs, "@default_class", "__custom_class");
  artm::TopicModel initial_model;
  initial_model.set_name(master_config.pwt_name());
  for (int i = 0; i < nTopics; ++i) {
    std::stringstream ss;
    ss << "Topic" << i;
    initial_model.add_topic_name(ss.str());
  }

  for (int i = 0; i < batch.token_size(); i++) {
    initial_model.add_token(batch.token(i));
    initial_model.add_class_id(batch.class_id(i));
    ::artm::FloatArray* token_weights = initial_model.add_token_weights();
    for (int topic_index = 0; topic_index < nTopics; ++topic_index) {
      token_weights->add_value(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));  // NOLINT
    }
  }

  std::vector<std::shared_ptr< ::artm::Batch>> batches;
  batches.push_back(std::make_shared< ::artm::Batch>(batch));

  ::artm::MasterModel master(master_config); ::artm::test::Api api(master);
  ::artm::MasterModel master3(master_config3); ::artm::test::Api api3(master3);
  ::artm::MasterModel master_reg(master_config_reg); ::artm::test::Api api_reg(master_reg);

  auto offlineArgs = api.Initialize(batches);
  auto offlineArgs2 = api3.Initialize(batches);
  auto offlineArgs_reg = api_reg.Initialize(batches);

  // Create model
  api.OverwriteModel(initial_model);
  api3.OverwriteModel(initial_model);
  api_reg.OverwriteModel(initial_model);

  // Index doc-token matrix
  int nIters = 5;
  ::artm::ThetaMatrix theta_matrix1_explicit, theta_matrix3_explicit;
  for (int iter = 0; iter < 5; ++iter) {
    if (iter == (nIters - 1)) {
      // Now we would like to verify that master_component.GetThetaMatrix gives the same result in two cases:
      // 1. Retrieving ThetaMatrix cached on the last iteration  (done in Processor::ThreadFunction() method)
      // 2. Explicitly getting ThateMatrix for the batch (done in Processor::FindThetaMatrix() method)
      // These results should be identical only if the same version of TopicModel is used in both cases.
      // This imply that we should cached theta matrix with GetThetaMatrix(batch) at the one-before-last iteration.
      // An alternative would be to not invoke model.Synchronize on the last iteration.
      ::artm::TransformMasterModelArgs transform_args;
      transform_args.set_theta_matrix_type(::artm::ThetaMatrixType_Dense);
      transform_args.add_batch_filename(batch.id());

      theta_matrix1_explicit = master.Transform(transform_args);
      theta_matrix3_explicit = master3.Transform(transform_args);
    }

    master.FitOfflineModel(offlineArgs);
    master3.FitOfflineModel(offlineArgs);
    master_reg.FitOfflineModel(offlineArgs);
  }

  ::artm::TopicModel topic_model1 = master.GetTopicModel();
  ::artm::TopicModel topic_model3 = master3.GetTopicModel();
  ::artm::TopicModel topic_model_reg = master_reg.GetTopicModel();

  ::artm::ThetaMatrix theta_matrix1 = master.GetThetaMatrix();
  ::artm::ThetaMatrix theta_matrix3 = master3.GetThetaMatrix();
  ::artm::ThetaMatrix theta_matrix_reg = master_reg.GetThetaMatrix();

  ShowTopicModel(topic_model1);
  ShowTopicModel(topic_model3);
  ShowTopicModel(topic_model_reg);

  ::artm::Matrix matrix_phi, matrix_theta;
  ::artm::TopicModel model_ex1 = master.GetTopicModel(&matrix_phi);
  ::artm::ThetaMatrix theta_ex1 = master.GetThetaMatrix(&matrix_theta);
  EXPECT_EQ(theta_ex1.item_weights_size(), 0);
  EXPECT_EQ(model_ex1.token_weights_size(), 0);
  ASSERT_EQ(matrix_phi.no_rows(), nTokens);
  ASSERT_EQ(matrix_phi.no_columns(), nTopics);
  ASSERT_EQ(matrix_theta.no_rows(), nDocs);
  ASSERT_EQ(matrix_theta.no_columns(), nTopics);
  for (int token_index = 0; token_index < nTokens; ++token_index) {
    for (int topic_index = 0; topic_index < nTopics; ++topic_index) {
      ASSERT_APPROX_EQ(matrix_phi(token_index, topic_index),
                       topic_model1.token_weights(token_index).value(topic_index));
    }
  }
  for (int topic_index = 0; topic_index < nTopics; ++topic_index) {
    for (int item_index = 0; item_index < nDocs; ++item_index) {
      ASSERT_APPROX_EQ(matrix_theta(item_index, topic_index),
                       theta_matrix1.item_weights(item_index).value(topic_index));
    }
  }

  // ToDo: validate matrix_phi and matrix_theta

  // ShowThetaMatrix(*theta_matrix1);
  // ShowThetaMatrix(*theta_matrix1_explicit);
  // ShowThetaMatrix(*theta_matrix3);
  // ShowThetaMatrix(*theta_matrix3_explicit);
  // ShowThetaMatrix(*theta_matrix_reg);  // <- 3 and 7 topics should be sparse in this matrix.

  float max_diff;
  // Compare consistency between Theta calculation in Processor::ThreadFunction() and Processor::FindThetaMatrix()
  EXPECT_TRUE(CompareThetaMatrices(theta_matrix1, theta_matrix1_explicit, &max_diff));
  EXPECT_LT(max_diff, 0.001);  // "theta_matrix1 == theta_matrix1_explicit");
  EXPECT_TRUE(CompareThetaMatrices(theta_matrix3, theta_matrix3_explicit, &max_diff));
  EXPECT_LT(max_diff, 0.001);  // "theta_matrix3 == theta_matrix3_explicit");

  // Verify that changing class_weight has an effect on the resulting model
  EXPECT_TRUE(CompareTopicModels(topic_model3, topic_model1, &max_diff));
  EXPECT_GT(max_diff, 0.001);  // topic_model3 != topic_model1

  EXPECT_TRUE(CompareThetaMatrices(theta_matrix3, theta_matrix1, &max_diff));
  EXPECT_GT(max_diff, 0.001);  // "theta_matrix3 != theta_matrix1");
}

// artm_tests.exe --gtest_filter=MultipleClasses.InitializeSomeModalities
TEST(MultipleClasses, InitializeSomeModalities) {
  int nTopics = 10;

  ::artm::MasterModelConfig master_config = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  master_config.add_class_id("@default_class");
  master_config.add_class_id("__custom_class");

  ::artm::DictionaryData d1;
  d1.set_name("d1");
  d1.add_class_id("@default_class"); d1.add_token("t1");
  d1.add_class_id("not present"); d1.add_token("t2");

  ::artm::DictionaryData d2; d2.set_name("d2");
  d2.add_class_id("not present"); d2.add_token("t2");

  ::artm::MasterModel mm(master_config);
  mm.CreateDictionary(d1);
  mm.CreateDictionary(d2);

  ::artm::InitializeModelArgs ia;
  ia.set_dictionary_name("d1");
  mm.InitializeModel(ia);

  auto tm = mm.GetTopicModel();

  ASSERT_EQ(tm.token_size(), 1);
  ASSERT_EQ(tm.token(0), "t1");

  ia.set_dictionary_name("d2");
  ia.set_model_name("m2");
  ASSERT_THROW(mm.InitializeModel(ia), ::artm::InvalidOperationException);
}

// artm_tests.exe --gtest_filter=MultipleClasses.ThrowIfNoTokensInEffect
TEST(MultipleClasses, ThrowIfNoTokensInEffect) {
  int nTokens = 60;
  int nDocs = 100;
  int nTopics = 10;

  ::artm::MasterModelConfig master_config = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  master_config.add_class_id("@default_class"); master_config.add_class_weight(0.5f);
  master_config.add_class_id("__custom_class"); master_config.add_class_weight(2.0f);

  ::artm::MasterModelConfig master_config_reg(master_config);

  // Generate doc-token matrix
  artm::Batch batch = ::artm::test::Helpers::GenerateBatch(nTokens, nDocs, "@default_class", "__custom_class");
  std::vector<std::shared_ptr< ::artm::Batch>> batches;
  batches.push_back(std::make_shared< ::artm::Batch>(batch));

  ::artm::MasterModel master(master_config);
  ::artm::test::Api api(master);
  auto offlineArgs = api.Initialize(batches);

  master_config.clear_class_id(); master_config.clear_class_weight();
  master_config.add_class_id("__unknown_class");
  master.Reconfigure(master_config);

  // Index doc-token matrix
  ASSERT_THROW(master.FitOfflineModel(offlineArgs), ::artm::InvalidOperationException);
}

void configureTopTokensScore(std::string score_name, std::string class_id, artm::MasterModelConfig* master_config) {
  ::artm::ScoreConfig score_config;
  ::artm::TopTokensScoreConfig top_tokens_config;
  top_tokens_config.set_num_tokens(4);
  if (!class_id.empty()) {
    top_tokens_config.set_class_id(class_id);
  }

  score_config.set_config(top_tokens_config.SerializeAsString());
  score_config.set_type(::artm::ScoreType_TopTokens);
  score_config.set_name(score_name);
  master_config->add_score_config()->CopyFrom(score_config);
}

void configureThetaSnippetScore(std::string score_name, int num_items, artm::MasterModelConfig* master_config) {
  ::artm::ScoreConfig score_config;
  ::artm::ThetaSnippetScoreConfig theta_snippet_config;
  theta_snippet_config.set_num_items(num_items);
  score_config.set_config(theta_snippet_config.SerializeAsString());
  score_config.set_type(::artm::ScoreType_ThetaSnippet);
  score_config.set_name(score_name);
  master_config->add_score_config()->CopyFrom(score_config);
}

void configureItemsProcessedScore(std::string score_name, artm::MasterModelConfig* master_config) {
  ::artm::ScoreConfig score_config;
  ::artm::ItemsProcessedScore items_processed_config;
  score_config.set_config(items_processed_config.SerializeAsString());
  score_config.set_type(::artm::ScoreType_ItemsProcessed);
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
  int nTokens = 60, nDocs = 100, nTopics = 10;

  ::artm::MasterModelConfig master_config = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);

  configureTopTokensScore("default_class", "", &master_config);
  configureTopTokensScore("tts_class_one", "class_one", &master_config);
  configureTopTokensScore("tts_class_two", "class_two", &master_config);
  configureThetaSnippetScore("theta_snippet", /*num_items = */ 5, &master_config);
  ::artm::test::Helpers::ConfigurePerplexityScore("perplexity", &master_config);
  configureItemsProcessedScore("items_processed", &master_config);

  master_config.add_class_id("class_one"); master_config.add_class_weight(2.0f);
  ::artm::MasterModel master(master_config);
  ::artm::test::Api api(master);

  master_config.add_class_id("class_two"); master_config.add_class_weight(0.5f);
  ::artm::MasterModel master2(master_config);
  ::artm::test::Api api2(master2);

  // Generate doc-token matrix
  artm::Batch batch = ::artm::test::Helpers::GenerateBatch(nTokens, nDocs, "class_one", "class_two");
  artm::DictionaryData dict = ::artm::test::Helpers::GenerateDictionary(nTokens, "class_one", "");
  std::vector<std::shared_ptr< ::artm::Batch>> batches;
  batches.push_back(std::make_shared< ::artm::Batch>(batch));

  auto offline_args = api.Initialize(batches, nullptr, nullptr, &dict);
  auto offline_args2 = api2.Initialize(batches);

  for (int iter = 0; iter < 5; ++iter) {
    master.FitOfflineModel(offline_args);
    master2.FitOfflineModel(offline_args2);
  }

  ::artm::TopicModel topic_model1 = master.GetTopicModel();
  ::artm::TopicModel topic_model2 = master2.GetTopicModel();
  EXPECT_EQ(topic_model1.token_size(), 30);
  EXPECT_EQ(topic_model2.token_size(), 60);
  // ShowTopicModel(*topic_model1);
  // ShowTopicModel(*topic_model2);

  ::artm::GetScoreValueArgs gs;
  gs.set_score_name("default_class"); EXPECT_EQ(master.GetScoreAs< ::artm::TopTokensScore>(gs).num_entries(), 0);
  gs.set_score_name("tts_class_one"); EXPECT_TRUE(master.GetScoreAs< ::artm::TopTokensScore>(gs).num_entries() > 0);
  gs.set_score_name("tts_class_two"); EXPECT_EQ(master.GetScoreAs< ::artm::TopTokensScore>(gs).num_entries(), 0);
  gs.set_score_name("default_class"); EXPECT_EQ(master2.GetScoreAs< ::artm::TopTokensScore>(gs).num_entries(), 0);
  gs.set_score_name("tts_class_one"); EXPECT_TRUE(master2.GetScoreAs< ::artm::TopTokensScore>(gs).num_entries() > 0);
  gs.set_score_name("tts_class_two"); EXPECT_TRUE(master2.GetScoreAs< ::artm::TopTokensScore>(gs).num_entries() > 0);

  gs.set_score_name("perplexity");
  float p1 = master.GetScoreAs< ::artm::PerplexityScore>(gs).value();
  float p2 = master2.GetScoreAs< ::artm::PerplexityScore>(gs).value();
  EXPECT_TRUE((p1 > 0) && (p2 > 0) && (p1 != p2));

  gs.set_score_name("theta_snippet");
  auto theta_snippet = master.GetScoreAs< ::artm::ThetaSnippetScore>(gs);
  EXPECT_EQ(theta_snippet.item_id_size(), 5);

  gs.set_score_name("items_processed");
  EXPECT_EQ(master.GetScoreAs< ::artm::ItemsProcessedScore>(gs).value(), nDocs);
}

void VerifySparseVersusDenseTopicModel(const ::artm::GetTopicModelArgs& args, ::artm::MasterModel* master) {
  ::artm::GetTopicModelArgs args_dense(args);
  args_dense.set_matrix_layout(artm::MatrixLayout_Dense);
  auto tm_dense = master->GetTopicModel(args_dense);

  ::artm::GetTopicModelArgs args_sparse(args);
  args_sparse.set_matrix_layout(artm::MatrixLayout_Sparse);
  auto tm_sparse = master->GetTopicModel(args_sparse);

  ::artm::GetTopicModelArgs args_all;
  auto tm_all = master->GetTopicModel(args_all);

  bool all_topics = args.topic_name_size() == 0;
  bool all_tokens = args.token_size() == 0;
  bool some_classes = all_tokens && (args.class_id_size() > 0);

  ASSERT_GT(tm_dense.topic_name_size(), 0);
  ASSERT_GT(tm_sparse.topic_name_size(), 0);
  ASSERT_GT(tm_dense.token_size(), 0);
  ASSERT_GT(tm_sparse.token_size(), 0);

  if (!all_topics) {
    for (int i = 0; i < tm_dense.topic_name_size(); ++i) {
      EXPECT_EQ(tm_dense.topic_name(i), args.topic_name(i));
    }
    for (int i = 0; i < tm_sparse.topic_name_size(); ++i) {
      EXPECT_EQ(tm_sparse.topic_name(i), args.topic_name(i));
    }
  }

  ASSERT_EQ(tm_sparse.token_size(), tm_dense.token_size());
  ASSERT_EQ(tm_sparse.token_weights_size(), tm_dense.token_weights_size());
  ASSERT_EQ(tm_sparse.class_id_size(), tm_dense.class_id_size());
  ASSERT_TRUE(tm_sparse.token_size() == tm_sparse.token_weights_size() &&
              tm_sparse.token_size() == tm_sparse.class_id_size());
  if (!all_tokens) {
    ASSERT_TRUE(tm_sparse.token_size() == args.token_size());
  }

  for (int i = 0; i < tm_sparse.token_size(); ++i) {
    EXPECT_EQ(tm_sparse.token(i), tm_dense.token(i));
    EXPECT_EQ(tm_sparse.class_id(i), tm_dense.class_id(i));
    if (!all_tokens) {
      EXPECT_EQ(tm_sparse.token(i), args.token(i));
      if (args.class_id_size() > 0) {
        EXPECT_EQ(tm_sparse.class_id(i), args.class_id(i));
      } else {
        EXPECT_EQ(tm_sparse.class_id(i), "@default_class");
      }
    }

    if (some_classes) {
      bool contains = false;
      for (int j = 0; j < args.class_id_size(); ++j) {
        if (args.class_id(j) == tm_sparse.class_id(i)) {
          contains = true;
        }
      }
      EXPECT_TRUE(contains);  // only return classes that had been requested
    }

    EXPECT_EQ(tm_dense.topic_indices_size(), 0);
    const ::artm::FloatArray& dense_topic = tm_dense.token_weights(i);
    const ::artm::FloatArray& sparse_topic = tm_sparse.token_weights(i);
    const ::artm::IntArray& sparse_topic_index = tm_sparse.topic_indices(i);
    ASSERT_EQ(sparse_topic.value_size(), sparse_topic_index.value_size());
    for (int j = 0; j < sparse_topic.value_size(); ++j) {
      int topic_index = sparse_topic_index.value(j);
      float value = sparse_topic.value(j);
      ASSERT_TRUE(topic_index >= 0 && topic_index <= tm_all.topic_name_size());
      EXPECT_TRUE(value >= args.eps());
      EXPECT_EQ(value, dense_topic.value(topic_index));
    }
  }
}

void VerifySparseVersusDenseThetaMatrix(const ::artm::GetThetaMatrixArgs& args, ::artm::MasterModel* master) {
  ::artm::GetThetaMatrixArgs args_dense(args);
  args_dense.set_matrix_layout(artm::MatrixLayout_Dense);
  auto tm_dense = master->GetThetaMatrix(args_dense);

  ::artm::GetThetaMatrixArgs args_sparse(args);
  args_sparse.set_matrix_layout(artm::MatrixLayout_Sparse);
  auto tm_sparse = master->GetThetaMatrix(args_sparse);

  auto tm_all = master->GetThetaMatrix();

  bool by_names = args.topic_name_size() > 0;

  ASSERT_EQ(tm_dense.num_topics(), tm_dense.topic_name_size());
  ASSERT_EQ(tm_sparse.num_topics(), tm_sparse.topic_name_size());
  ASSERT_GT(tm_dense.num_topics(), 0);
  ASSERT_GT(tm_sparse.num_topics(), 0);
  ASSERT_GT(tm_dense.item_id_size(), 0);
  ASSERT_GT(tm_sparse.item_id_size(), 0);

  if (by_names) {
    ASSERT_EQ(tm_dense.num_topics(), args.topic_name_size());
    for (int i = 0; i < tm_dense.num_topics(); ++i) {
      EXPECT_EQ(tm_dense.topic_name(i), args.topic_name(i));
    }
  } else {
    ASSERT_EQ(tm_dense.num_topics(), tm_all.num_topics());
  }

  ASSERT_EQ(tm_sparse.num_topics(), tm_all.num_topics());
  for (int i = 0; i < tm_sparse.num_topics(); ++i) {
    EXPECT_EQ(tm_sparse.topic_name(i), tm_all.topic_name(i));
  }

  ASSERT_EQ(tm_sparse.item_id_size(), tm_dense.item_id_size());
  ASSERT_EQ(tm_sparse.item_weights_size(), tm_dense.item_weights_size());
  ASSERT_EQ(tm_sparse.item_title_size(), tm_dense.item_title_size());
  ASSERT_TRUE(tm_sparse.item_id_size() == tm_sparse.item_weights_size() &&
              tm_sparse.item_id_size() == tm_sparse.item_title_size());

  for (int i = 0; i < tm_sparse.item_id_size(); ++i) {
    EXPECT_EQ(tm_sparse.item_id(i), tm_dense.item_id(i));
    EXPECT_EQ(tm_sparse.item_title(i), tm_dense.item_title(i));
    EXPECT_EQ(tm_dense.topic_indices_size(), 0);
    const ::artm::FloatArray& dense_topic = tm_dense.item_weights(i);
    const ::artm::FloatArray& sparse_topic = tm_sparse.item_weights(i);
    const ::artm::IntArray& sparse_topic_index = tm_sparse.topic_indices(i);
    ASSERT_EQ(sparse_topic.value_size(), sparse_topic_index.value_size());
    for (int j = 0; j < sparse_topic.value_size(); ++j) {
      int topic_index = sparse_topic_index.value(j);
      float value = sparse_topic.value(j);
      ASSERT_TRUE(topic_index >= 0 && topic_index <= tm_all.num_topics());
      EXPECT_TRUE(value >= args.eps());
      EXPECT_EQ(value, dense_topic.value(topic_index));
    }
  }
}

// artm_tests.exe --gtest_filter=MultipleClasses.GetTopicModel
TEST(MultipleClasses, GetTopicModel) {
  int nTokens = 60, nDocs = 100, nTopics = 10;
  ::artm::MasterModelConfig master_config;
  master_config.set_pwt_name("pwt");

  for (int i = 0; i < nTopics; ++i) {
    std::stringstream ss;
    ss << "Topic" << i;
    master_config.add_topic_name(ss.str());
  }

  master_config.add_class_id("class_one"); master_config.add_class_weight(1.0f);
  master_config.add_class_id("class_two"); master_config.add_class_weight(1.0f);
  master_config.set_cache_theta(true);
  ::artm::MasterModel master(master_config);
  ::artm::test::Api api(master);

  // Generate doc-token matrix
  artm::Batch batch = ::artm::test::Helpers::GenerateBatch(nTokens, nDocs, "class_one", "class_two");
  std::vector<std::shared_ptr< ::artm::Batch>> batches;
  batches.push_back(std::make_shared< ::artm::Batch>(batch));
  auto offline_args = api.Initialize(batches);
  offline_args.set_num_collection_passes(5);
  master.FitOfflineModel(offline_args);

  ::artm::GetTopicModelArgs args;
  args.set_eps(0.05f);
  VerifySparseVersusDenseTopicModel(args, &master);

  for (int i = 0; i < nTopics; i += 2) {
    args.add_topic_name(master_config.topic_name(i));
  }
  VerifySparseVersusDenseTopicModel(args, &master);

  args.add_class_id("class_two");
  VerifySparseVersusDenseTopicModel(args, &master);

  args.add_token("token1");  // class_two
  args.add_token("token0"); args.add_class_id("class_one");
  VerifySparseVersusDenseTopicModel(args, &master);

  ::artm::GetThetaMatrixArgs args_theta;
  args_theta.set_eps(0.05f);
  VerifySparseVersusDenseThetaMatrix(args_theta, &master);
}
