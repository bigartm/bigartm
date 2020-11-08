// Copyright 2017, Additive Regularization of Topic Models.

#include <fstream>  // NOLINT
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid.hpp"
#include "boost/uuid/uuid_generators.hpp"
#include "boost/uuid/uuid_io.hpp"

#include "gtest/gtest.h"

#include "google/protobuf/util/json_util.h"

#include "artm/cpp_interface.h"
#include "artm/core/common.h"

void GenerateBatches(std::vector< ::artm::Batch>* batches, ::artm::DictionaryData* dictionary = nullptr) {
  int nBatches = 10;
  int nItemsPerBatch = 5;
  int nTokens = 40;

  // Generate global dictionary
  std::vector<std::string> tokens;
  for (int i = 0; i < nTokens; i++) {
    std::stringstream str;
    str << "token" << i;

    auto token = str.str();
    tokens.push_back(token);
    if (dictionary != nullptr) {
      dictionary->add_token(token);
    }
  }

  // Keep batch.token empty; batch.item.field.token_id point straight to global dictionary
  int itemId = 0;
  for (int iBatch = 0; iBatch < nBatches; ++iBatch) {
    ::artm::Batch batch;
    batch.set_id(boost::lexical_cast<std::string>(boost::uuids::random_generator()()));

    for (const auto& token : tokens) {
      batch.add_token(token);
    }

    for (int iItem = 0; iItem < nItemsPerBatch; ++iItem) {
      artm::Item* item = batch.add_item();
      item->set_id(itemId++);
      for (int iToken = 0; iToken < nTokens; ++iToken) {
        // Add each third token (randomly)
        if (rand() % 3 == 0) {  // NOLINT
          item->add_token_id(iToken);
          item->add_transaction_start_index(item->transaction_start_index_size());
          item->add_token_weight(1.0);
        }
      }
      item->add_transaction_start_index(item->transaction_start_index_size());
    }

    batches->push_back(batch);
  }
}

std::vector<std::string> getTopicNames() {
  std::vector<std::string> topic_names;
  int nTopics = 10;
  for (int i = 0; i < nTopics; i++) {
    std::stringstream str;
    str << "token" << i;
    topic_names.push_back(str.str());
  }
  return topic_names;
}

void describeTheta(const ::artm::ThetaMatrix& theta, int first_items) {
  std::cout << "Total items: " << theta.item_id_size() << "\n";
  for (int item = 0; item < std::min(first_items, theta.item_id_size()); ++item) {
    std::cout << "Item#" << item << " topics: ";
    for (int i = 0; i < theta.item_weights(item).value_size(); ++i)
      std::cout << theta.item_weights(item).value(i) << " ";
    std::cout << "\n";
  }
  if (first_items < theta.item_id_size())
    std::cout << "...\n";
}

void describeTopTokensScore(const ::artm::TopTokensScore& top_tokens) {
  /*
  message TopTokensScore{
    optional int32 num_entries;
    repeated string topic_name;
    repeated string token;
    repeated float weight;
  }
  */
  for (int i = 0; i < top_tokens.num_entries(); ++i) {
    bool is_new_topic = (i == 0 || top_tokens.topic_name(i) != top_tokens.topic_name(i - 1));
    if (is_new_topic) {
      std::cout << std::endl << top_tokens.topic_name(i) << ": ";
    } else {
      std::cout << ", ";
    }
    std::cout << top_tokens.token(i) << "(" << std::setprecision(3) << top_tokens.weight(i) << ")";
  }
  std::cout << std::endl;
}

// Static object to pass topic model through memory between two test cases (Fit and TransformAfterOverwrite)
static std::shared_ptr< ::artm::TopicModel> topic_model;

// To run this particular test:
// artm_tests.exe --gtest_filter=Supcry.Fit
TEST(Supcry, Fit) {
  // Step 1. Configure and create MasterModel
  ::artm::MasterModelConfig config;

  // Add topic names (this steps defines how many topics it will be in the topic model)
  for (const auto& topic_name : getTopicNames()) {
    config.add_topic_name(topic_name);
  }

  ::artm::ScoreConfig* score_config = config.add_score_config();
  score_config->set_type(::artm::ScoreType_Perplexity);
  score_config->set_name("Perplexity");
  score_config->set_config(::artm::PerplexityScoreConfig().SerializeAsString());

  ::artm::ScoreConfig* tts_config = config.add_score_config();
  tts_config->set_type(::artm::ScoreType_TopTokens);
  tts_config->set_name("TopTokens");
  tts_config->set_config(::artm::TopTokensScoreConfig().SerializeAsString());

  ::artm::RegularizerConfig* reg_theta = config.add_regularizer_config();
  reg_theta->set_type(::artm::RegularizerType_SmoothSparseTheta);
  reg_theta->set_tau(-0.2);
  reg_theta->set_name("SparseTheta");
  reg_theta->set_config(::artm::SmoothSparseThetaConfig().SerializeAsString());

  ::artm::MasterModel master_model(config);

  // Step 2. Generate dictionary and batches
  std::vector< ::artm::Batch> batches;
  ::artm::DictionaryData dictionary_data;
  GenerateBatches(&batches, &dictionary_data);

  // Step 3. Import batches into BigARTM memory
  ::artm::ImportBatchesArgs import_batches_args;
  for (auto& batch : batches) {
    import_batches_args.add_batch()->CopyFrom(batch);
  }
  master_model.ImportBatches(import_batches_args);

  // Step 4. Import dictionary into BigARTM memory
  dictionary_data.set_name("dictionary");
  master_model.CreateDictionary(dictionary_data);

  // Step 5. Initialize model
  ::artm::InitializeModelArgs initialize_model_args;
  initialize_model_args.set_dictionary_name(dictionary_data.name());
  master_model.InitializeModel(initialize_model_args);

  // Step 6. Fit topic model using offline algorithm
  ::artm::GetScoreValueArgs get_score_args;
  for (int pass = 0; pass < 4; pass++) {
    master_model.FitOfflineModel(::artm::FitOfflineMasterModelArgs());

    get_score_args.set_score_name(score_config->name());
    artm::PerplexityScore perplexity_score = master_model.GetScoreAs< ::artm::PerplexityScore>(get_score_args);
    std::cout << "Perplexity@" << pass << " = " << perplexity_score.value() << "\n";
  }

  // Step 7. Show top tokens score
  get_score_args.set_score_name(tts_config->name());
  auto top_tokens = master_model.GetScoreAs< ::artm::TopTokensScore>(get_score_args);
  describeTopTokensScore(top_tokens);

  // Step 8. Export topic model
  ::artm::ExportModelArgs export_model_args;
  export_model_args.set_file_name("artm_model.bin");

  try { boost::filesystem::remove("artm_model.bin"); }
  catch (...) { }
  master_model.ExportModel(export_model_args);

  // Step 9. Memory export
  ::artm::GetTopicModelArgs get_model_args;
  topic_model = std::make_shared< ::artm::TopicModel>(master_model.GetTopicModel(get_model_args));
}

// To run this particular test:
// artm_tests.exe --gtest_filter=Supcry.Transform
TEST(Supcry, TransformAfterImport) {
  // Step 1. Configure and create MasterModel
  ::artm::MasterModelConfig config;

  // Add topic names (this steps defines how many topics it will be in the topic model)
  for (const auto& topic_name : getTopicNames()) {
    config.add_topic_name(topic_name);
  }

  ::artm::MasterModel master_model(config);

  // Step 2. Generate batches
  std::vector< ::artm::Batch> batches;
  GenerateBatches(&batches);

  // Step 3. Import topic model
  ::artm::ImportModelArgs import_model_args;
  import_model_args.set_file_name("artm_model.bin");
  master_model.ImportModel(import_model_args);

  // Step 4. Find theta matrix
  ::artm::TransformMasterModelArgs transform_args;
  for (const auto& batch : batches) {
    transform_args.add_batch()->CopyFrom(batch);
  }
  ::artm::ThetaMatrix theta = master_model.Transform(transform_args);

  describeTheta(theta, 5);
}

// To run this particular test:
// artm_tests.exe --gtest_filter=Supcry.Transform
TEST(Supcry, TransformAfterOverwrite) {
  // Step 1. Configure and create MasterModel
  ::artm::MasterModelConfig config;

  // Add topic names (this steps defines how many topics it will be in the topic model)
  for (const auto& topic_name : getTopicNames()) {
    config.add_topic_name(topic_name);
  }

  ::artm::MasterModel master_model(config);

  // Step 2. Generate batches
  std::vector< ::artm::Batch> batches;
  GenerateBatches(&batches);

  // Step 3. Import topic model
  topic_model->set_name("garbage");  // to test ArtmOverwriteTopicModelNamed
  std::string blob;
  if (ArtmProtobufMessageFormatIsJson()) {
    ::google::protobuf::util::MessageToJsonString(*topic_model, &blob);
  } else {
    topic_model->SerializeToString(&blob);
  }
  ArtmOverwriteTopicModelNamed(master_model.id(), blob.size(), &*(blob.begin()), /*name=*/ nullptr);

  // Step 4. Find theta matrix
  ::artm::TransformMasterModelArgs transform_args;
  for (const auto& batch : batches) {
    transform_args.add_batch()->CopyFrom(batch);
  }
  ::artm::ThetaMatrix theta = master_model.Transform(transform_args);

  describeTheta(theta, 5);
}

// To run this particular test:
// artm_tests.exe --gtest_filter=Supcry.FitFromDiskFolder
TEST(Supcry, FitFromDiskFolder) {
  // Step 1. Configure and create MasterModel
  ::artm::MasterModelConfig config;
  for (const auto& topic_name : getTopicNames()) {
    config.add_topic_name(topic_name);
  }

  ::artm::ScoreConfig* score_config = config.add_score_config();
  score_config->set_type(::artm::ScoreType_Perplexity);
  score_config->set_name("Perplexity");
  score_config->set_config(::artm::PerplexityScoreConfig().SerializeAsString());
  ::artm::MasterModel master_model(config);

  // Step 2. Generate batches and save them to disk
  std::string batch_folder = "./batch_folder";
  try { boost::filesystem::remove_all(batch_folder); }
  catch (...) { }
  boost::filesystem::create_directory(batch_folder);

  std::vector< ::artm::Batch> batches;
  ::artm::DictionaryData dictionary_data;
  GenerateBatches(&batches, &dictionary_data);
  for (auto& batch : batches) {
    auto batch_path = boost::filesystem::path(batch_folder) / (batch.id() + ".batch");
    std::ofstream fout(batch_path.string().c_str(), std::ofstream::binary);
    ASSERT_TRUE(batch.SerializeToOstream(&fout));
  }

  // Step 3. Import dictionary into BigARTM memory
  dictionary_data.set_name("dictionary");
  master_model.CreateDictionary(dictionary_data);

  // Step 5. Initialize model
  ::artm::InitializeModelArgs initialize_model_args;
  initialize_model_args.set_dictionary_name(dictionary_data.name());
  master_model.InitializeModel(initialize_model_args);

  // Step 6. Fit topic model using offline algorithm
  for (int pass = 0; pass < 4; pass++) {
    ::artm::FitOfflineMasterModelArgs fit_offline_args;
    fit_offline_args.set_batch_folder(batch_folder);
    master_model.FitOfflineModel(fit_offline_args);

    ::artm::GetScoreValueArgs get_score_args;
    get_score_args.set_score_name(score_config->name());
    artm::PerplexityScore perplexity_score = master_model.GetScoreAs< ::artm::PerplexityScore>(get_score_args);
    std::cout << "Perplexity@" << pass << " = " << perplexity_score.value() << "\n";
  }
}
