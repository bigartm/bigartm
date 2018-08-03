// Copyright 2017, Additive Regularization of Topic Models.

#include "boost/thread.hpp"
#include "gtest/gtest.h"

#include "boost/filesystem.hpp"
#include "glog/logging.h"

#include "google/protobuf/util/json_util.h"

#include "artm/c_interface.h"
#include "artm/cpp_interface.h"
#include "artm/core/exceptions.h"
#include "artm/core/common.h"
#include "artm/core/protobuf_helpers.h"
#include "artm/core/phi_matrix_operations.h"

#include "artm/core/helpers.h"

#include "artm_tests/test_mother.h"
#include "artm_tests/api.h"

namespace fs = boost::filesystem;
namespace pb = google::protobuf;

TEST(CppInterface, Canary) {
}

TEST(CppInterface, Version) {
  std::string version(ArtmGetVersion());
  EXPECT_GT(version.size(), 0);
}

void RunBasicTest(bool serialize_as_json) {
  if (serialize_as_json) {
    ArtmSetProtobufMessageFormatToJson();
  } else {
    ArtmSetProtobufMessageFormatToBinary();
  }

  artm::ConfigureLoggingArgs log_args;
  log_args.set_minloglevel(2);
  artm::ConfigureLogging(log_args);
  EXPECT_EQ(FLAGS_minloglevel, log_args.minloglevel());

  std::string target_path = artm::test::Helpers::getUniqueString();

  ::artm::MasterModelConfig master_config;
  master_config.set_cache_theta(true);
  master_config.set_disk_cache_path(".");
  master_config.set_pwt_name("model_config1");

  master_config.add_topic_name("first topic");
  master_config.add_topic_name("second topic");
  master_config.add_topic_name("third topic");
  master_config.add_topic_name("4th topic");
  master_config.add_topic_name("5th topic");
  const int nTopics = master_config.topic_name_size();

  ::artm::ScoreConfig* score_config = master_config.add_score_config();
  if (serialize_as_json) {
    pb::util::MessageToJsonString(::artm::PerplexityScoreConfig(), score_config->mutable_config_json());
  } else {
    score_config->set_config(::artm::PerplexityScoreConfig().SerializeAsString());
  }
  score_config->set_type(::artm::ScoreType_Perplexity);
  score_config->set_name("PerplexityScore");

  // check log level
  EXPECT_EQ(FLAGS_minloglevel, log_args.minloglevel());
  log_args.set_minloglevel(1);
  artm::ConfigureLogging(log_args);
  EXPECT_EQ(FLAGS_minloglevel, log_args.minloglevel());

  ::artm::RegularizerConfig* reg_decor_config = master_config.add_regularizer_config();
  artm::DecorrelatorPhiConfig decor_config;
  reg_decor_config->set_name("decorrelator");
  reg_decor_config->set_type(artm::RegularizerType_DecorrelatorPhi);
  if (serialize_as_json) {
    pb::util::MessageToJsonString(decor_config, reg_decor_config->mutable_config_json());
  } else {
    reg_decor_config->set_config(decor_config.SerializeAsString());
  }
  reg_decor_config->set_tau(1.0);

  // Create master component
  artm::MasterModel master_component(master_config);
  ::artm::test::Api api(master_component);

  EXPECT_GT(master_component.info().score_size(), 1);
  EXPECT_EQ(master_component.info().score(0).name(), "PerplexityScore");
  EXPECT_EQ(master_component.info().regularizer_size(), 1);

  // Load doc-token matrix
  int nTokens = 10;
  int nDocs = 15;

  std::vector< std::shared_ptr< ::artm::Batch>> batches;
  batches.push_back(std::make_shared< ::artm::Batch>());
  artm::Batch& batch = *batches.back();
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
    for (int iToken = 0; iToken < nTokens; ++iToken) {
      item->add_token_id(iToken);
      item->add_transaction_start_index(item->transaction_start_index_size());
      item->add_token_weight(static_cast<float>(iDoc + iToken + 1));
    }
    item->add_transaction_start_index(item->transaction_start_index_size());
  }

  EXPECT_EQ(batch.item().size(), nDocs);
  for (int i = 0; i < batch.item().size(); i++) {
    EXPECT_EQ(batch.item(i).token_id_size(), nTokens);
  }

  // Index doc-token matrix
  ::artm::FitOfflineMasterModelArgs offline_args = api.Initialize(batches);

  artm::TopicModel topic_model;
  double expected_normalizer = 0.0;
  float previous_perplexity = 0.0f;
  for (int iter = 0; iter < 5; ++iter) {
    master_component.FitOfflineModel(offline_args);
    {
      topic_model = master_component.GetTopicModel();
    }

    ::artm::GetScoreValueArgs get_score_args;
    get_score_args.set_score_name("PerplexityScore");
    auto perplexity = master_component.GetScoreAs< ::artm::PerplexityScore>(get_score_args);
    if (serialize_as_json) {
      EXPECT_FALSE(master_component.GetScore(get_score_args).data_json().empty());
    }

    if (iter > 0) {
      EXPECT_EQ(perplexity.value(), previous_perplexity);
    }

    ::artm::TransformMasterModelArgs transform_args;
    transform_args.add_batch_filename(batch.id());
    transform_args.set_theta_matrix_type(::artm::ThetaMatrixType_Cache);
    master_component.Transform(transform_args);
    auto perplexity2 = master_component.GetScoreAs< ::artm::PerplexityScore>(get_score_args);
    previous_perplexity = perplexity2.value();

    if (iter == 1) {
      expected_normalizer = perplexity.normalizer();
      EXPECT_GT(expected_normalizer, 0);
    } else if (iter >= 2) {
      // Verify that normalizer does not grow starting from second iteration.
      // This confirms that the Instance::ForceResetScores() function works as expected.
      EXPECT_EQ(perplexity.normalizer(), expected_normalizer);
    }
  }

  int nUniqueTokens = nTokens;
  ASSERT_EQ(nUniqueTokens, topic_model.token_size());
  auto first_token_topics = topic_model.token_weights(0);
  EXPECT_EQ(first_token_topics.value_size(), nTopics);

  {
    ::artm::ThetaMatrix theta_matrix = master_component.GetThetaMatrix();

    EXPECT_EQ(theta_matrix.item_id_size(), nDocs);
    EXPECT_EQ(theta_matrix.item_title_size(), nDocs);
    EXPECT_EQ(theta_matrix.num_topics(), nTopics);
    for (int item_index = 0; item_index < theta_matrix.item_id_size(); ++item_index) {
      EXPECT_EQ(theta_matrix.item_id(item_index), 666 + item_index);
      EXPECT_EQ(theta_matrix.item_title(item_index), item_title[item_index]);
      const ::artm::FloatArray& weights = theta_matrix.item_weights(item_index);
      ASSERT_EQ(weights.value_size(), nTopics);
      float sum = 0;
      for (int topic_index = 0; topic_index < weights.value_size(); ++topic_index) {
        float weight = weights.value(topic_index);
        EXPECT_GT(weight, 0);
        sum += weight;
      }

      EXPECT_LE(std::abs(sum - 1), 0.001);
    }

    ::artm::GetThetaMatrixArgs get_theta_args;
    get_theta_args.add_topic_name("third topic");
    get_theta_args.add_topic_name("4th topic");
    ::artm::ThetaMatrix theta_matrix23 = master_component.GetThetaMatrix(get_theta_args);
    EXPECT_EQ(theta_matrix23.item_id_size(), nDocs);
    EXPECT_EQ(theta_matrix23.num_topics(), 2);
    for (int item_index = 0; item_index < theta_matrix23.item_id_size(); ++item_index) {
      const ::artm::FloatArray& weights23 = theta_matrix23.item_weights(item_index);
      const ::artm::FloatArray& weights = theta_matrix.item_weights(item_index);
      ASSERT_EQ(weights23.value_size(), 2);
      EXPECT_EQ(weights23.value(0), weights.value(2));
      EXPECT_EQ(weights23.value(1), weights.value(3));
    }
  }

  // Test overwrite topic model
  artm::TopicModel new_topic_model;
  new_topic_model.set_name("model2_name");
  new_topic_model.mutable_topic_name()->CopyFrom(master_component.config().topic_name());
  new_topic_model.add_token("my overwritten token");
  new_topic_model.add_token("my overwritten token2");
  auto weights = new_topic_model.add_token_weights();
  auto weights2 = new_topic_model.add_token_weights();
  for (int i = 0; i < nTopics; ++i) {
    weights->add_value(static_cast<float>(i));
    weights2->add_value(static_cast<float>(nTopics - i));
  }
  api.OverwriteModel(new_topic_model);

  ::artm::NormalizeModelArgs normalize_args;
  normalize_args.set_nwt_source_name("model2_name");
  normalize_args.set_pwt_target_name("model2_name");
  api.NormalizeModel(normalize_args);

  {
    artm::GetTopicModelArgs args;
    args.set_model_name(new_topic_model.name());
    args.add_token("my overwritten token");
    args.add_class_id("@default_class");
    args.add_token("my overwritten token2");
    args.add_class_id("@default_class");

    auto new_topic_model2 = master_component.GetTopicModel(args);
    ASSERT_EQ(new_topic_model2.token_size(), 2);
    EXPECT_EQ(new_topic_model2.token(0), "my overwritten token");
    EXPECT_EQ(new_topic_model2.token(1), "my overwritten token2");
    for (int i = 0; i < nTopics; ++i) {
      EXPECT_FLOAT_EQ(
        new_topic_model2.token_weights(0).value(i),
        static_cast<float>(i) / static_cast<float>(nTopics));

      EXPECT_FLOAT_EQ(
        new_topic_model2.token_weights(1).value(i),
        1.0f - static_cast<float>(i) / static_cast<float>(nTopics));
    }
  }

  // Test dictionaries and InitializeModel
  ::artm::DictionaryData dict_config;
  dict_config.set_name("My dictionary");
  dict_config.add_token("my_tok_1");
  dict_config.add_token("my_tok_2");
  dict_config.add_token("my_tok_3");
  master_component.CreateDictionary(dict_config);
  ::artm::InitializeModelArgs init_model_args;
  init_model_args.set_model_name("model3_name");
  init_model_args.set_dictionary_name(dict_config.name());
  init_model_args.mutable_topic_name()->CopyFrom(master_component.config().topic_name());
  master_component.InitializeModel(init_model_args);

  artm::GetTopicModelArgs args;
  args.set_model_name(init_model_args.model_name());
  auto new_topic_model3 = master_component.GetTopicModel(args);
  ASSERT_EQ(new_topic_model3.token_size(), 3);
  ASSERT_EQ(new_topic_model3.token(0), "my_tok_1");
  ASSERT_EQ(new_topic_model3.token(1), "my_tok_2");
  ASSERT_EQ(new_topic_model3.token(2), "my_tok_3");

  try { boost::filesystem::remove_all(target_path); }
  catch (...) { }
}

// artm_tests.exe --gtest_filter=CppInterface.BasicTest
TEST(CppInterface, BasicTest) {
  bool wasJson = ArtmProtobufMessageFormatIsJson();
  try { RunBasicTest(false); } catch (...) { }
  wasJson ? ArtmSetProtobufMessageFormatToJson() : ArtmSetProtobufMessageFormatToBinary();
}

TEST(CppInterface, BasicTestJson) {
  bool wasJson = ArtmProtobufMessageFormatIsJson();
  RunBasicTest(true);
  wasJson ? ArtmSetProtobufMessageFormatToJson() : ArtmSetProtobufMessageFormatToBinary();
}

// artm_tests.exe --gtest_filter=CppInterface.BasicTrasactionTest
TEST(CppInterface, BasicTrasactionTest) {
  std::string target_folder = artm::test::Helpers::getUniqueString();
  try {
    ::artm::MasterModelConfig master_config;
    master_config.set_cache_theta(true);
    master_config.set_disk_cache_path(".");
    master_config.set_pwt_name("pwt");

    int nTopics = 5;
    for (int i = 0; i < nTopics; ++i) {
      std::stringstream ss;
      ss << "topic_" << i;
      master_config.add_topic_name(ss.str());
    }

    // Create master component
    artm::MasterModel master_component(master_config);
    ::artm::test::Api api(master_component);

    ::artm::CollectionParserConfig config;
    config.set_format(::artm::CollectionParserConfig_CollectionFormat_VowpalWabbit);
    config.set_target_folder(target_folder);
    config.set_docword_file_path("../../../test_data/vw_transaction_data_extended.txt");
    config.set_num_items_per_batch(2);

    ::artm::ParseCollection(config);

    // Test InitializeModel and model extraction
    artm::GatherDictionaryArgs dict_args;
    dict_args.add_batch_path(target_folder);
    dict_args.set_dictionary_target_name("dict");
    dict_args.set_data_path(target_folder);
    master_component.GatherDictionary(dict_args);

    ::artm::InitializeModelArgs init_model_args;
    init_model_args.set_model_name("model");
    init_model_args.set_dictionary_name(dict_args.dictionary_target_name());
    init_model_args.mutable_topic_name()->CopyFrom(master_component.config().topic_name());
    master_component.InitializeModel(init_model_args);

    auto test_func = [](artm::MasterModel& master_component, const artm::InitializeModelArgs& init_model_args) {
      artm::GetTopicModelArgs args;
      args.set_model_name(init_model_args.model_name());
      auto model_1 = master_component.GetTopicModel(args);

      std::unordered_map<artm::core::ClassId, float> normalizer_keys;
      for (int i = 0; i < model_1.token_size(); ++i) {
        float value = model_1.token_weights(i).value(0);
        normalizer_keys[model_1.class_id(i)] += value;
      }

      ASSERT_EQ(normalizer_keys.size(), 4);
      for (const auto& nk : normalizer_keys) {
        ASSERT_FLOAT_EQ(nk.second, 1.0);
      }
    };

    test_func(master_component, init_model_args);

    artm::ExportModelArgs exp_args;
    fs::path export_filename = (fs::path(target_folder) / fs::path(artm::test::Helpers::getUniqueString() + ".model"));
    exp_args.set_file_name(export_filename.string());
    exp_args.set_model_name(init_model_args.model_name());
    master_component.ExportModel(exp_args);

    artm::ImportModelArgs imp_args;
    imp_args.set_file_name(exp_args.file_name());
    imp_args.set_model_name(init_model_args.model_name());
    master_component.ImportModel(imp_args);

    test_func(master_component, init_model_args);

    artm::TopicModel new_topic_model;
    new_topic_model.set_name("model2_name");
    new_topic_model.mutable_topic_name()->CopyFrom(master_component.config().topic_name());

    new_topic_model.add_token("token_x");
    new_topic_model.add_class_id("class_5");

    new_topic_model.add_token("token_y");
    new_topic_model.add_class_id("class_5");

    auto weights = new_topic_model.add_token_weights();
    auto weights2 = new_topic_model.add_token_weights();
    for (int i = 0; i < nTopics; ++i) {
      weights->add_value(static_cast<float>(i));
      weights2->add_value(static_cast<float>(nTopics - i));
    }
    api.OverwriteModel(new_topic_model);

    ::artm::NormalizeModelArgs normalize_args;
    normalize_args.set_nwt_source_name(new_topic_model.name());
    normalize_args.set_pwt_target_name(new_topic_model.name());
    api.NormalizeModel(normalize_args);

    {
      artm::GetTopicModelArgs args;
      args.set_model_name(new_topic_model.name());
      new_topic_model.add_token("token_x");
      new_topic_model.add_class_id("class_5");

      new_topic_model.add_token("token_y");
      new_topic_model.add_class_id("class_5");

      auto new_topic_model2 = master_component.GetTopicModel(args);

      ASSERT_EQ(new_topic_model2.token_size(), 2);
      EXPECT_EQ(new_topic_model2.token(0), "token_x");
      EXPECT_EQ(new_topic_model2.token(1), "token_y");
      for (int i = 0; i < nTopics; ++i) {
        EXPECT_FLOAT_EQ(
          new_topic_model2.token_weights(0).value(i),
          static_cast<float>(i) / static_cast<float>(nTopics));
          EXPECT_FLOAT_EQ(
            new_topic_model2.token_weights(1).value(i),
            1.0f - static_cast<float>(i) / static_cast<float>(nTopics));
      }
    }
  }
  catch (...) {
    try { boost::filesystem::remove_all(target_folder); }
    catch (...) {}
  }
}

// artm_tests.exe --gtest_filter=CppInterface.ProcessBatchesApi
TEST(CppInterface, ProcessBatchesApi) {
  int nTopics = 17;
  int nBatches = 5;

  std::string target_folder = artm::test::Helpers::getUniqueString();
  ASSERT_TRUE(boost::filesystem::create_directory(target_folder));

  std::vector<std::shared_ptr< ::artm::Batch>> batches =
    ::artm::test::TestMother::GenerateBatches(nBatches, 50);
  ASSERT_EQ(batches.size(), nBatches);

  artm::MasterModelConfig master_config;
  master_config.set_pwt_name("pwt0");
  artm::ScoreConfig* score_config = master_config.add_score_config();
  score_config->set_name("Perplexity");
  score_config->set_type(artm::ScoreType_Perplexity);
  ::artm::PerplexityScoreConfig perplexity_score_config;
  score_config->set_config(perplexity_score_config.SerializeAsString());
  artm::MasterModel master(master_config);
  artm::test::Api api(master);

  master.DisposeDictionary(std::string());  // Dispose all dictionaries (if any leaked from previous tests)

  artm::ImportBatchesArgs import_batches_args;
  artm::GatherDictionaryArgs gather_args;
  for (auto& batch_path : batches) {
    import_batches_args.add_batch()->CopyFrom(*batch_path);
    gather_args.add_batch_path(batch_path->id());
  }
  master.ImportBatches(import_batches_args);

  gather_args.set_dictionary_target_name("gathered_dictionary");
  master.GatherDictionary(gather_args);

  artm::InitializeModelArgs initialize_model_args;
  initialize_model_args.set_dictionary_name("gathered_dictionary");
  for (int i = 0; i < nTopics; ++i)
    initialize_model_args.add_topic_name("Topic" + boost::lexical_cast<std::string>(i));
  initialize_model_args.set_model_name("pwt0");
  master.InitializeModel(initialize_model_args);
  ::artm::MasterComponentInfo master_info = master.info();
  ASSERT_EQ(master_info.model_size(), 1);  // "pwt0"
  EXPECT_EQ(master_info.model(0).name(), "pwt0");
  EXPECT_EQ(master_info.model(0).num_topics(), nTopics);

  ::artm::GetTopicModelArgs get_topic_model_args;
  get_topic_model_args.set_model_name("pwt0");
  ::artm::TopicModel pwt_model = master.GetTopicModel(get_topic_model_args);
  ASSERT_EQ(pwt_model.topic_name_size(), nTopics);

  // Test export and import of new-style models
  artm::ExportModelArgs export_model_args;
  export_model_args.set_model_name(pwt_model.name());

  fs::path export_filename = (fs::path(target_folder) / fs::path(artm::test::Helpers::getUniqueString() + ".model"));
  export_model_args.set_model_name("pwt0");
  export_model_args.set_file_name(export_filename.string());
  artm::ImportModelArgs import_model_args;
  import_model_args.set_model_name("import_pwt");
  import_model_args.set_file_name(export_model_args.file_name());

  master.ExportModel(export_model_args);
  master.ImportModel(import_model_args);
  ASSERT_EQ(master.info().model_size(), 2);  // "pwt0", "import_pwt"
  bool ok2 = false;
  get_topic_model_args.set_model_name("import_pwt");
  ::artm::test::Helpers::CompareTopicModels(master.GetTopicModel(), master.GetTopicModel(get_topic_model_args), &ok2);
  if (!ok2) {
    std::cout << "Exported topic model:\n"
      << ::artm::test::Helpers::DescribeTopicModel(master.GetTopicModel());
    std::cout << "Imported topic model:\n"
      << ::artm::test::Helpers::DescribeTopicModel(master.GetTopicModel(get_topic_model_args));
  }

  master.DisposeModel("import_pwt");
  ASSERT_EQ(master.info().model_size(), 1);  // "pwt0"
  /////////////////////////////////////////////

  artm::ProcessBatchesArgs process_batches_args;
  process_batches_args.mutable_batch_filename()->CopyFrom(gather_args.batch_path());
  process_batches_args.set_nwt_target_name("nwt_hat");

  artm::NormalizeModelArgs normalize_model_args;
  normalize_model_args.set_pwt_target_name("pwt");
  normalize_model_args.set_nwt_source_name("nwt_hat");

  ::artm::PerplexityScore perplexity_score;
  ::artm::GetScoreValueArgs score_args;
  score_args.set_score_name("Perplexity");
  for (int i = 0; i < 10; ++i) {  // 10 iterations
    process_batches_args.set_pwt_source_name(i == 0 ? "pwt0" : "pwt");
    process_batches_args.set_theta_matrix_type(artm::ThetaMatrixType_Dense);
    api.ClearScoreCache(::artm::ClearScoreCacheArgs());
    artm::ThetaMatrix result = api.ProcessBatches(process_batches_args);
    perplexity_score = master.GetScoreAs< ::artm::PerplexityScore>(score_args);
    EXPECT_EQ(result.num_topics(), nTopics);
    EXPECT_EQ(result.item_id_size(), nBatches);  // assuming that each batch has just one document
    api.NormalizeModel(normalize_model_args);
  }

  ASSERT_EQ(master.info().model_size(), 3);  // "pwt0", "pwt", "nwt_hat"
  EXPECT_NE(perplexity_score.value(), 0.0);

  ::artm::FitOfflineMasterModelArgs offline_args;
  offline_args.mutable_batch_filename()->CopyFrom(process_batches_args.batch_filename());
  for (int i = 0; i < 10; ++i)
    master.FitOfflineModel(offline_args);

  auto perplexity_score2 = master.GetScoreAs< ::artm::PerplexityScore>(score_args);
  ASSERT_APPROX_EQ(perplexity_score2.value(), perplexity_score.value());

  bool ok = false;
  get_topic_model_args.set_model_name("pwt");
  ::artm::test::Helpers::CompareTopicModels(master.GetTopicModel(get_topic_model_args), master.GetTopicModel(), &ok);

  if (!ok) {
    std::cout << "New        -tuned topic model:\n"
      << ::artm::test::Helpers::DescribeTopicModel(master.GetTopicModel(get_topic_model_args));
    std::cout << "MasterModel-tuned topic model:\n"
      << ::artm::test::Helpers::DescribeTopicModel(master.GetTopicModel());
  }

  ::artm::DictionaryData dict_config;
  dict_config.set_name("My dictionary");
  dict_config.add_token("my_tok_1");
  dict_config.add_class_id("@default_class");
  dict_config.add_token_df(1.0f);
  dict_config.add_token_tf(2.0f);
  master.CreateDictionary(dict_config);
  master_info = master.info();
  ASSERT_EQ(master_info.dictionary_size(), 2);
  EXPECT_EQ(master_info.dictionary(0).num_entries(), 1);
  ASSERT_GE(master_info.model_size(), 1);
  for (auto& model_info : master_info.model())
    EXPECT_GT(model_info.byte_size(), 0);

  {
    artm::MasterModel master_clone(api.Duplicate(::artm::DuplicateMasterComponentArgs()));
    ::artm::MasterComponentInfo clone_master_info = master_clone.info();
    ASSERT_EQ(clone_master_info.model_size(), 4);  // "pwt", "nwt_hat"; "pwt0" is not cloned (old-style model)
    ASSERT_EQ(clone_master_info.dictionary_size(), 2);
    EXPECT_EQ(clone_master_info.dictionary(0).num_entries(), 1);
    ASSERT_EQ(clone_master_info.score_size(), master_info.score_size());
    ASSERT_EQ(clone_master_info.regularizer_size(), master_info.regularizer_size());

    ::artm::test::Helpers::CompareTopicModels(master_clone.GetTopicModel(),
                                              master.GetTopicModel(), &ok);
    ASSERT_TRUE(ok);
    ArtmDisposeMasterComponent(master_clone.id());
  }

  // Verify that we may call ProcessBatches without nwt_target
  process_batches_args.clear_nwt_target_name();
  ::artm::ThetaMatrix result_1 = api.ProcessBatches(process_batches_args);
  EXPECT_NE(result_1.item_id_size(), 0);

  // Dummy test to verify we can merge models
  ::artm::MergeModelArgs merge_model_args;
  merge_model_args.add_nwt_source_name("pwt"); merge_model_args.add_source_weight(1.0f);
  merge_model_args.add_nwt_source_name("pwt0"); merge_model_args.add_source_weight(1.0f);
  merge_model_args.set_nwt_target_name("nwt_merge");
  api.MergeModel(merge_model_args);
  get_topic_model_args.set_model_name("nwt_merge");
  ::artm::TopicModel nwt_merge = master.GetTopicModel(get_topic_model_args);
  ASSERT_EQ(nwt_merge.num_topics(), nTopics);

  // Dummy test to verify we can regularize models
  auto config = master.config();
  ::artm::RegularizerConfig* sparse_phi_config = config.add_regularizer_config();
  sparse_phi_config->set_name("sparse_phi");
  sparse_phi_config->set_type(::artm::RegularizerType_SmoothSparsePhi);
  sparse_phi_config->set_config(::artm::SmoothSparsePhiConfig().SerializeAsString());
  sparse_phi_config->set_tau(1.0);
  master.Reconfigure(config);

  ::artm::RegularizeModelArgs regularize_model_args;
  regularize_model_args.set_rwt_target_name("rwt");
  regularize_model_args.set_pwt_source_name("pwt");
  regularize_model_args.set_nwt_source_name("nwt_hat");
  ::artm::RegularizerSettings* regularizer_settings = regularize_model_args.add_regularizer_settings();
  regularizer_settings->set_name("sparse_phi");
  regularizer_settings->set_tau(-0.5);
  api.RegularizeModel(regularize_model_args);
  get_topic_model_args.set_model_name("rwt");
  ::artm::TopicModel rwt = master.GetTopicModel(get_topic_model_args);
  ASSERT_EQ(rwt.topic_name_size(), nTopics);

  // Test to verify Ptdw extraction
  process_batches_args.set_theta_matrix_type(artm::ThetaMatrixType_SparsePtdw);
  artm::ThetaMatrix theta_matrix = api.ProcessBatches(process_batches_args);

  const int expected_combined_items_length = 154;
  ASSERT_EQ(theta_matrix.item_id_size(), expected_combined_items_length);
  ASSERT_EQ(theta_matrix.topic_indices_size(), expected_combined_items_length);
  ASSERT_EQ(theta_matrix.item_weights_size(), expected_combined_items_length);
  for (int index = 0; index < theta_matrix.item_id_size(); ++index) {
    ASSERT_GE(theta_matrix.topic_indices(index).value_size(), 0);
    ASSERT_GE(theta_matrix.item_weights(index).value_size(), 0);
    ASSERT_EQ(theta_matrix.topic_indices(index).value_size(), theta_matrix.item_weights(index).value_size());
  }

  try { boost::filesystem::remove_all(target_folder); }
  catch (...) { }
}

// artm_tests.exe --gtest_filter=CppInterface.AttachModel
TEST(CppInterface, AttachModel) {
  int nTopics = 17, nBatches = 5, nTokens = 50;
  std::string target_folder = artm::test::Helpers::getUniqueString();
  auto batches = ::artm::test::TestMother::GenerateBatches(nBatches, nTokens);
  artm::MasterModelConfig master_config = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  artm::MasterModel master(master_config);
  ::artm::test::Api api(master);
  auto offlineArgs = api.Initialize(batches);

  ::artm::AttachModelArgs attach_args;
  ::artm::GetTopicModelArgs get_model_args;
  attach_args.set_model_name(master_config.pwt_name());
  get_model_args.set_model_name(master_config.pwt_name());
  ::artm::Matrix attached_pwt;
  api.AttachTopicModel(attach_args, &attached_pwt);
  ::artm::TopicModel pwt0_model = master.GetTopicModel(get_model_args);
  ASSERT_EQ(attached_pwt.no_rows(), pwt0_model.token_size());
  ASSERT_EQ(attached_pwt.no_columns(), pwt0_model.num_topics());

  ::artm::MergeModelArgs merge_model_args;
  merge_model_args.add_nwt_source_name(master_config.pwt_name()); merge_model_args.add_source_weight(1.0f);
  merge_model_args.set_nwt_target_name("nwt_merge");
  api.MergeModel(merge_model_args);
  attach_args.set_model_name("nwt_merge");
  get_model_args.set_model_name("nwt_merge");
  ::artm::Matrix attached_nwt_merge;
  api.AttachTopicModel(attach_args, &attached_nwt_merge);
  ::artm::TopicModel nwt_merge_model = master.GetTopicModel(get_model_args);
  ASSERT_EQ(attached_nwt_merge.no_rows(), nwt_merge_model.token_size());
  ASSERT_EQ(attached_nwt_merge.no_columns(), nwt_merge_model.num_topics());

  // Verify that it is possible to modify the attached matrix
  for (int token_index = 0; token_index < nwt_merge_model.token_size(); ++token_index) {
    for (int topic_index = 0; topic_index < nwt_merge_model.num_topics(); ++topic_index) {
      ASSERT_APPROX_EQ(attached_nwt_merge(token_index, topic_index),
                       nwt_merge_model.token_weights(token_index).value(topic_index));
      attached_nwt_merge(token_index, topic_index) = 2.0f * token_index + 3.0f * topic_index;
    }
  }

  ::artm::TopicModel updated_model = master.GetTopicModel(get_model_args);
  for (int token_index = 0; token_index < nwt_merge_model.token_size(); ++token_index) {
    for (int topic_index = 0; topic_index < nwt_merge_model.num_topics(); ++topic_index) {
      ASSERT_APPROX_EQ(updated_model.token_weights(token_index).value(topic_index),
                       2.0f * token_index + 3.0f * topic_index);
    }
  }

  {
    // Verify that we can clone attached model
    bool ok = false;
    artm::MasterModel master_clone(api.Duplicate(::artm::DuplicateMasterComponentArgs()));
    ::artm::test::Helpers::CompareTopicModels(master_clone.GetTopicModel(get_model_args),
      master.GetTopicModel(get_model_args), &ok);
    ASSERT_TRUE(ok);
    ArtmDisposeMasterComponent(master_clone.id());
  }

  // Good practice is to dispose model once its attachment is gone.
  master.DisposeModel(master_config.pwt_name());
  master.DisposeModel("nwt_merge");


  try { boost::filesystem::remove_all(target_folder); }
  catch (...) { }
}

// artm_tests.exe --gtest_filter=CppInterface.AsyncProcessBatches
TEST(CppInterface, AsyncProcessBatches) {
  int nTopics = 17, nBatches = 5, nTokens = 50;
  auto batches = ::artm::test::TestMother::GenerateBatches(nBatches, nTokens);
  artm::MasterModelConfig master_config = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  artm::MasterModel master(master_config);
  ::artm::test::Api api(master);
  auto offlineArgs = api.Initialize(batches);

  std::vector<int> operation_ids;
  for (unsigned i = 0; i < offlineArgs.batch_filename_size(); ++i) {
    const std::string& batch_name = offlineArgs.batch_filename(i);
    artm::ProcessBatchesArgs process_batches_args;
    process_batches_args.add_batch_filename(batch_name);
    process_batches_args.set_pwt_source_name(std::string(master_config.pwt_name()));
    process_batches_args.set_nwt_target_name(std::string("nwt_hat") + boost::lexical_cast<std::string>(i));
    process_batches_args.set_theta_matrix_type(::artm::ThetaMatrixType_None);
    operation_ids.push_back(api.AsyncProcessBatches(process_batches_args));
  }

  for (unsigned i = 0; i < operation_ids.size(); ++i) {
    api.AwaitOperation(operation_ids[i]);

    ::artm::MergeModelArgs merge_model_args;
    std::string name = std::string("nwt_hat") + boost::lexical_cast<std::string>(i);
    merge_model_args.add_nwt_source_name("nwt_merge"); merge_model_args.add_source_weight(1.0f);
    merge_model_args.add_nwt_source_name(name); merge_model_args.add_source_weight(1.0f);
    merge_model_args.set_nwt_target_name("nwt_merge");
    api.MergeModel(merge_model_args);
  }

  artm::NormalizeModelArgs normalize_model_args;
  normalize_model_args.set_pwt_target_name("pwt");
  normalize_model_args.set_nwt_source_name("nwt_merge");
  api.NormalizeModel(normalize_model_args);
}

// artm_tests.exe --gtest_filter=CppInterface.Dictionaries
TEST(CppInterface, Dictionaries) {
  int nBatches = 5;
  std::string target_folder = artm::test::Helpers::getUniqueString();
  ::artm::test::TestMother::GenerateBatches(nBatches, 50, target_folder);
  artm::MasterModelConfig master_config;
  artm::MasterModel master(master_config);

  // Gather
  artm::GatherDictionaryArgs gather_args;
  gather_args.set_data_path(target_folder);
  gather_args.set_dictionary_target_name("gathered_dictionary");
  master.GatherDictionary(gather_args);

  ::artm::GetDictionaryArgs get_dict;
  get_dict.set_dictionary_name("gathered_dictionary");
  auto dictionary = master.GetDictionary(get_dict);
  ASSERT_EQ(dictionary.token_size(), 50);
  ASSERT_GT(dictionary.token_df(0), 0);
  ASSERT_GT(dictionary.token_tf(0), 0);
  ASSERT_GT(dictionary.token_value(0), 0);

  // Filter
  artm::FilterDictionaryArgs filter_args;
  filter_args.set_dictionary_name("gathered_dictionary");
  filter_args.set_dictionary_target_name("filtered_dictionary");
  filter_args.set_max_df(4);
  master.FilterDictionary(filter_args);

  get_dict.set_dictionary_name("filtered_dictionary");
  dictionary = master.GetDictionary(get_dict);
  ASSERT_EQ(dictionary.token_size(), 32);
  ASSERT_GT(dictionary.token_df(0), 0);
  ASSERT_GT(dictionary.token_tf(0), 0);
  ASSERT_GT(dictionary.token_value(0), 0);

  // Export
  artm::ExportDictionaryArgs export_args;
  export_args.set_file_name(artm::test::Helpers::getUniqueString() + ".dict");
  export_args.set_dictionary_name("filtered_dictionary");
  master.ExportDictionary(export_args);

  // Import
  artm::ImportDictionaryArgs import_args;
  import_args.set_file_name(export_args.file_name());
  import_args.set_dictionary_name("imported_dictionary");
  master.ImportDictionary(import_args);

  get_dict.set_dictionary_name("imported_dictionary");
  dictionary = master.GetDictionary(get_dict);
  ASSERT_EQ(dictionary.token_size(), 32);
  ASSERT_GT(dictionary.token_df(0), 0);
  ASSERT_GT(dictionary.token_tf(0), 0);
  ASSERT_GT(dictionary.token_value(0), 0);

  try { boost::filesystem::remove_all(target_folder); }
  catch (...) { }

  try { boost::filesystem::remove(import_args.file_name()); }
  catch (...) { }
}

// artm_tests.exe --gtest_filter=ProtobufMessages.Json
TEST(ProtobufMessages, Json) {
  ::artm::MasterModelConfig config, config2;
  config.set_num_processors(12);
  std::string json;
  ASSERT_EQ(::google::protobuf::util::MessageToJsonString(config, &json),
            ::google::protobuf::util::Status::OK);
  ASSERT_EQ(::google::protobuf::util::JsonStringToMessage("{num_processors:12}", &config2),
            ::google::protobuf::util::Status::OK);
  ASSERT_EQ(config.num_processors(), config2.num_processors());
}

// artm_tests.exe --gtest_filter=CppInterface.TransactionDictionaries
TEST(CppInterface, TransactionDictionaries) {
  std::string target_folder = artm::test::Helpers::getUniqueString();

  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_CollectionFormat_VowpalWabbit);
  config.set_target_folder(target_folder);
  config.set_docword_file_path((::artm::test::Helpers ::getTestDataDir() /
                                "vw_transaction_data_extended.txt").string());
  config.set_num_items_per_batch(2);

  ::artm::ParseCollection(config);

  artm::MasterModelConfig master_config;
  artm::MasterModel master(master_config);

  // Gather
  artm::GatherDictionaryArgs gather_args;
  gather_args.set_data_path(target_folder);
  gather_args.set_dictionary_target_name("gathered_dictionary");
  master.GatherDictionary(gather_args);

  ::artm::GetDictionaryArgs get_dict;
  get_dict.set_dictionary_name("gathered_dictionary");
  ASSERT_EQ(master.GetDictionary(get_dict).token_size(), 8);

  // Export & Import
  artm::ExportDictionaryArgs export_args;
  export_args.set_file_name(artm::test::Helpers::getUniqueString() + ".dict");
  export_args.set_dictionary_name("gathered_dictionary");
  master.ExportDictionary(export_args);

  artm::ImportDictionaryArgs import_args;
  import_args.set_file_name(export_args.file_name());
  import_args.set_dictionary_name("imported_dictionary");
  master.ImportDictionary(import_args);

  get_dict.set_dictionary_name("imported_dictionary");
  ASSERT_EQ(master.GetDictionary(get_dict).token_size(), 8);

  // Get and Create
  artm::GetDictionaryArgs get_args;
  get_args.set_dictionary_name("imported_dictionary");
  auto data = master.GetDictionary(get_args);
  ASSERT_EQ(data.token_size(), 8);

  data.set_name("created_dictionary");
  master.CreateDictionary(data);

  get_args.set_dictionary_name("created_dictionary");
  data = master.GetDictionary(get_args);
  ASSERT_EQ(data.token_size(), 8);

  try { boost::filesystem::remove_all(target_folder); }
  catch (...) {}

  try { boost::filesystem::remove(import_args.file_name()); }
  catch (...) {}
}

// artm_tests.exe --gtest_filter=CppInterface.ReconfigureTopics
TEST(CppInterface, ReconfigureTopics) {
  ::artm::MasterModelConfig config;
  config.add_topic_name("t1"); config.add_topic_name("t2"); config.add_topic_name("t3");
  ::artm::DictionaryData dict; dict.add_token("token"); dict.set_name("d");

  ::artm::MasterModel mm(config);
  mm.CreateDictionary(dict);

  ::artm::InitializeModelArgs init; init.set_dictionary_name("d");
  mm.InitializeModel(init);
  auto m1 = mm.GetTopicModel();
  ASSERT_TRUE(::artm::core::repeated_field_equals(m1.topic_name(), config.topic_name()));

  config.clear_topic_name();
  config.add_topic_name("t3"); config.add_topic_name("t1"); config.add_topic_name("t4");
  mm.ReconfigureTopicName(config);
  auto m2 = mm.GetTopicModel();
  ASSERT_TRUE(::artm::core::repeated_field_equals(m2.topic_name(), config.topic_name()));
  ASSERT_EQ(m2.token_weights(0).value(0), m1.token_weights(0).value(2));  // "t3"
  ASSERT_EQ(m2.token_weights(0).value(1), m1.token_weights(0).value(0));  // "t1"
  ASSERT_EQ(m2.token_weights(0).value(2), 0);  // "t4" (new topic)

  ::artm::MergeModelArgs merge;
  merge.add_topic_name("t4");  // used just to provide the set of tokens
  merge.add_nwt_source_name(m1.name());
  merge.set_nwt_target_name("tmp");
  mm.MergeModel(merge);

  init.clear_dictionary_name();
  init.set_model_name("tmp");
  mm.InitializeModel(init);
  ::artm::GetTopicModelArgs get_model; get_model.set_model_name("tmp");
  auto m3 = mm.GetTopicModel(get_model);
  ASSERT_TRUE(::artm::core::repeated_field_equals(m3.topic_name(), merge.topic_name()));
  ASSERT_NE(m3.token_weights(0).value(0), 0.0f);

  merge.clear_topic_name();
  merge.clear_nwt_source_name();
  merge.add_nwt_source_name(m1.name());
  merge.add_nwt_source_name("tmp");
  merge.set_nwt_target_name(m1.name());
  mm.MergeModel(merge);
  auto m4 = mm.GetTopicModel();
  ASSERT_TRUE(::artm::core::repeated_field_equals(m4.topic_name(), config.topic_name()));
  ASSERT_EQ(m4.token_weights(0).value(0), m2.token_weights(0).value(0));  // t3, from m2
  ASSERT_EQ(m4.token_weights(0).value(1), m2.token_weights(0).value(1));  // t1, from m2
  ASSERT_EQ(m4.token_weights(0).value(2), m3.token_weights(0).value(0));  // t4, from m3
}

// artm_tests.exe --gtest_filter=CppInterface.MergeModelWithDictionary
TEST(CppInterface, MergeModelWithDictionary) {
  ::artm::MasterModelConfig config;
  config.add_topic_name("t1");
  ::artm::DictionaryData dict1; dict1.set_name("d1"); dict1.add_token("t1"); dict1.add_token("t2");
  ::artm::DictionaryData dict2; dict2.set_name("d2"); dict2.add_token("t3"); dict2.add_token("t1");
  ::artm::DictionaryData dict3; dict3.set_name("d3"); dict3.add_token("t1"); dict3.add_token("t4");
  dict3.add_token("t2");

  ::artm::MasterModel mm(config);
  mm.CreateDictionary(dict1);
  mm.CreateDictionary(dict2);
  mm.CreateDictionary(dict3);

  ::artm::InitializeModelArgs init;
  init.set_dictionary_name("d1"); init.set_model_name("m1"); mm.InitializeModel(init);
  init.set_dictionary_name("d2"); init.set_model_name("m2"); mm.InitializeModel(init);

  ::artm::GetTopicModelArgs get_model;
  get_model.set_model_name("m1"); auto m1 = mm.GetTopicModel(get_model);
  get_model.set_model_name("m2"); auto m2 = mm.GetTopicModel(get_model);

  ::artm::MergeModelArgs merge;
  merge.add_nwt_source_name("m1");
  merge.add_nwt_source_name("m2");
  merge.set_nwt_target_name("m");
  merge.set_dictionary_name("d3");
  mm.MergeModel(merge);
  get_model.set_model_name("m"); auto m = mm.GetTopicModel(get_model);

  ASSERT_EQ(m.token_size(), 3);
  ASSERT_EQ(m.token(0), "t1");
  ASSERT_EQ(m.token(1), "t4");
  ASSERT_EQ(m.token(2), "t2");

  ASSERT_EQ(m.token_weights(0).value(0), m1.token_weights(0).value(0) + m2.token_weights(1).value(0));
  ASSERT_EQ(m.token_weights(1).value(0), 0.0f);
  ASSERT_EQ(m.token_weights(2).value(0), m1.token_weights(1).value(0));
}
