#include <ctime>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
using namespace std;

#include "boost/filesystem.hpp"
using namespace boost::filesystem;

#include "boost/timer/timer.hpp"

#include "artm/cpp_interface.h"
#include "artm/messages.pb.h"
#include "glog/logging.h"
using namespace artm;

int countFilesInDirectory(std::string root, std::string ext) {
  int retval = 0;
  if (boost::filesystem::exists(root) && boost::filesystem::is_directory(root)) {
    boost::filesystem::recursive_directory_iterator it(root);
    boost::filesystem::recursive_directory_iterator endit;
    while(it != endit) {
      if (boost::filesystem::is_regular_file(*it) && it->path().extension() == ext) {
        retval++;
      }
      ++it;
    }
  }
  return retval;
}

void proc(int argc, char * argv[], int processors_count, int instance_size) {
  std::string batches_disk_path = "batches";
  std::string docword_file = "../../../datasets/docword.kos.txt";
  std::string vocab_file = "../../../datasets/vocab.kos.txt";
  std::string dictionary_file = "kos.dictionary";
  int topics_count = 16;

  // Recommended values for decorrelator_tau are as follows:
  // kos - 700000, nips - 200000.
  float decorrelator_tau = 200000;
  // float dirichlet_tau = -100;

  // instance_size = 0 stands for "connect to external node_controller process",
  // instance_size = 1 stands for "local modus operandi",
  // instance_size = 2 or higher defines the number of node controllers to create in this process,
  //                   used in the same way as if they were on remote nodes.
  bool is_network_mode = (instance_size != 1);

  MasterComponentConfig master_config;
  std::vector<std::shared_ptr<::artm::NodeController>> node_controller;
  if (is_network_mode) {
    for (int port = 5556; port < 5556 + instance_size; ++port) {
      ::artm::NodeControllerConfig node_config;

      std::stringstream port_str;
      port_str << port;
      node_config.set_create_endpoint(std::string("tcp://*:") + port_str.str());
      node_controller.push_back(std::make_shared<::artm::NodeController>(node_config));
      master_config.add_node_connect_endpoint(std::string("tcp://localhost:") + port_str.str());
    }

    if (instance_size == 0) {
      master_config.add_node_connect_endpoint("tcp://localhost:5556");
    }
  }

  master_config.set_processors_count(processors_count);
  batches_disk_path = (current_path() / path(batches_disk_path)).string();

  int batch_files_count = countFilesInDirectory(batches_disk_path, ".batch");
  std::shared_ptr<DictionaryConfig> unique_tokens;
  if (batch_files_count == 0) {
    ::artm::CollectionParserConfig collection_parser_config;
    collection_parser_config.set_format(CollectionParserConfig_Format_BagOfWordsUci);
    collection_parser_config.set_docword_file_path(docword_file);
    collection_parser_config.set_vocab_file_path(vocab_file);
    collection_parser_config.set_dictionary_file_name(dictionary_file);
    collection_parser_config.set_target_folder(batches_disk_path);
    unique_tokens = ::artm::ParseCollection(collection_parser_config);

    std::cout << "OK.\n";
  } else {
    std::cout << "Found " << batch_files_count << " batches in folder '"
              << batches_disk_path << "', will use them.\n";

    unique_tokens = ::artm::LoadDictionary((path(batches_disk_path) / dictionary_file).string());
  }
  
  master_config.set_disk_path(batches_disk_path);
  if (is_network_mode) {
    master_config.set_modus_operandi(MasterComponentConfig_ModusOperandi_Network);
    master_config.set_create_endpoint("tcp://*:5555");
    master_config.set_connect_endpoint("tcp://localhost:5555");
  } else {
    master_config.set_modus_operandi(MasterComponentConfig_ModusOperandi_Local);
    master_config.set_cache_theta(true);
  }

  ::artm::ScoreConfig score_config;
  ::artm::PerplexityScoreConfig perplexity_config;
  perplexity_config.set_stream_name("test_stream");
  score_config.set_config(perplexity_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_Perplexity);
  score_config.set_name("test_perplexity");
  master_config.add_score_config()->CopyFrom(score_config);

  perplexity_config.set_stream_name("train_stream");
  score_config.set_config(perplexity_config.SerializeAsString());
  score_config.set_name("train_perplexity");
  master_config.add_score_config()->CopyFrom(score_config);

  ::artm::SparsityThetaScoreConfig sparsity_theta_config;
  sparsity_theta_config.set_stream_name("test_stream");
  score_config.set_config(sparsity_theta_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_SparsityTheta);
  score_config.set_name("test_sparsity_theta");
  master_config.add_score_config()->CopyFrom(score_config);

  sparsity_theta_config.set_stream_name("train_stream");
  score_config.set_config(sparsity_theta_config.SerializeAsString());
  score_config.set_name("train_sparsity_theta");
  master_config.add_score_config()->CopyFrom(score_config);

  ::artm::SparsityPhiScoreConfig sparsity_phi_config;
  score_config.set_config(sparsity_phi_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_SparsityPhi);
  score_config.set_name("sparsity_phi");
  master_config.add_score_config()->CopyFrom(score_config);

  ::artm::ItemsProcessedScoreConfig items_processed_config;
  items_processed_config.set_stream_name("test_stream");
  score_config.set_config(items_processed_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_ItemsProcessed);
  score_config.set_name("test_items_processed");
  master_config.add_score_config()->CopyFrom(score_config);

  items_processed_config.set_stream_name("train_stream");
  score_config.set_config(items_processed_config.SerializeAsString());
  score_config.set_name("train_items_processed");
  master_config.add_score_config()->CopyFrom(score_config);

  ::artm::TopTokensScoreConfig top_tokens_config;
  top_tokens_config.set_num_tokens(6);
  score_config.set_config(top_tokens_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_TopTokens);
  score_config.set_name("top_tokens");
  master_config.add_score_config()->CopyFrom(score_config);

  ::artm::ThetaSnippetScoreConfig theta_snippet_config;
  theta_snippet_config.set_stream_name("train_stream");
  for (int id = 0; id < 7; ++id) theta_snippet_config.add_item_id(id);
  score_config.set_config(theta_snippet_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_ThetaSnippet);
  score_config.set_name("train_theta_snippet");
  master_config.add_score_config()->CopyFrom(score_config);

  ::artm::TopicKernelScoreConfig topic_kernel_config;
  std::string tr = topic_kernel_config.SerializeAsString();
  score_config.set_config(topic_kernel_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_TopicKernel);
  score_config.set_name("topic_kernel");
  master_config.add_score_config()->CopyFrom(score_config);

  // MasterProxyConfig master_proxy_config;
  // master_proxy_config.set_node_connect_endpoint("tcp://localhost:5555");
  // master_proxy_config.mutable_config()->CopyFrom(master_config);
  // master_proxy_config.set_communication_timeout(50000);
  // MasterComponent master_component(master_proxy_config);

  MasterComponent master_component(master_config);

  // Configure train and test streams
  Stream train_stream, test_stream;
  train_stream.set_name("train_stream");
  train_stream.set_type(Stream_Type_ItemIdModulus);
  train_stream.set_modulus(10);
  for (int i = 0; i <= 8; ++i) {
    train_stream.add_residuals(i);
  }

  test_stream.set_name("test_stream");
  test_stream.set_type(Stream_Type_ItemIdModulus);
  test_stream.set_modulus(10);
  test_stream.add_residuals(9);

  master_component.AddStream(train_stream);
  master_component.AddStream(test_stream);

  RegularizerConfig regularizer_config;
  std::string regularizer_decor_phi_name = "regularizer_decor_phi";
  regularizer_config.set_name(regularizer_decor_phi_name);
  regularizer_config.set_type(::artm::RegularizerConfig_Type_DecorrelatorPhi);
  regularizer_config.set_config(::artm::DecorrelatorPhiConfig().SerializeAsString());
  Regularizer decorrelator_phi_regularizer(master_component, regularizer_config);

  std::string regularizer_dirichlet_phi_name = "regularizer_dirichlet_phi";
  regularizer_config.set_name(regularizer_dirichlet_phi_name);
  regularizer_config.set_type(::artm::RegularizerConfig_Type_DirichletPhi);
  regularizer_config.set_config(::artm::DirichletPhiConfig().SerializeAsString());
  Regularizer dirichlet_phi_regularizer(master_component, regularizer_config);

  // Create model
  int nTopics = (topics_count == 0) ? atoi(argv[3]) : topics_count;
  ModelConfig model_config;
  model_config.set_topics_count(nTopics);
  model_config.set_inner_iterations_count(10);
  model_config.set_stream_name("train_stream");
  model_config.set_reuse_theta(true);
  model_config.set_name("15081980-90a7-4767-ab85-7cb551c39339");
  model_config.add_regularizer_name(regularizer_decor_phi_name);
  model_config.add_regularizer_tau(decorrelator_tau);
  //model_config.add_regularizer_name(regularizer_dirichlet_phi_name);
  //model_config.add_regularizer_tau(dirichlet_tau);
  model_config.add_score_name("test_perplexity");
  model_config.add_score_name("train_perplexity");
  model_config.add_score_name("test_sparsity_theta");
  model_config.add_score_name("train_sparsity_theta");
  model_config.add_score_name("sparsity_phi");
  model_config.add_score_name("test_items_processed");
  model_config.add_score_name("train_items_processed");
  model_config.add_score_name("top_tokens");
  model_config.add_score_name("train_theta_snippet");
  model_config.add_score_name("topic_kernel");

  Model model(master_component, model_config);

  // Overwrite topic model with well-known "initial topic model"
  TopicModel initial_topic_model;
  initial_topic_model.set_name(model_config.name());
  initial_topic_model.set_topics_count(nTopics);
  for (int token_index = 0; token_index < unique_tokens->entry_size(); ++token_index) {
    std::string token = unique_tokens->entry(token_index).key_token();
    initial_topic_model.add_token(token);
    artm::FloatArray* weights = initial_topic_model.add_token_weights();
    for (int topic_index = 0; topic_index < nTopics; ++topic_index) {
      weights->add_value((float) rand() / (float)RAND_MAX);
    }
  }

  model.Overwrite(initial_topic_model);

  boost::timer::cpu_timer timer;

  std::shared_ptr<TopicModel> topic_model;
  std::shared_ptr<PerplexityScore> test_perplexity, train_perplexity;
  std::shared_ptr<SparsityThetaScore> test_sparsity_theta, train_sparsity_theta;
  std::shared_ptr<SparsityPhiScore> sparsity_phi;
  std::shared_ptr<ItemsProcessedScore> test_items_processed, train_items_processed;
  std::shared_ptr<TopTokensScore> top_tokens;
  std::shared_ptr<ThetaSnippetScore> train_theta_snippet;
  std::shared_ptr<TopicKernelScore> topic_kernel;

  for (int iter = 0; iter < 10; ++iter) {
    master_component.InvokeIteration(1);
    master_component.WaitIdle(120000);
    model.InvokePhiRegularizers();

    topic_model = master_component.GetTopicModel(model);
    test_perplexity = master_component.GetScoreAs<::artm::PerplexityScore>(model, "test_perplexity");
    train_perplexity = master_component.GetScoreAs<::artm::PerplexityScore>(model, "train_perplexity");
    test_sparsity_theta = master_component.GetScoreAs<::artm::SparsityThetaScore>(model, "test_sparsity_theta");
    train_sparsity_theta = master_component.GetScoreAs<::artm::SparsityThetaScore>(model, "train_sparsity_theta");
    sparsity_phi = master_component.GetScoreAs<::artm::SparsityPhiScore>(model, "sparsity_phi");
    test_items_processed = master_component.GetScoreAs<::artm::ItemsProcessedScore>(model, "test_items_processed");
    train_items_processed = master_component.GetScoreAs<::artm::ItemsProcessedScore>(model, "train_items_processed");
    topic_kernel = master_component.GetScoreAs<::artm::TopicKernelScore>(model, "topic_kernel");

    std::cout << "Iter #" << (iter + 1) << ": "
              << "\n\t#Tokens = "  << topic_model->token_size() << ", "
              << "\n\tTest perplexity = " << test_perplexity->value() << ", "
              << "\n\tTrain perplexity = " << train_perplexity->value() << ", "
              << "\n\tTest spatsity theta = " << test_sparsity_theta->value() << ", "
              << "\n\tTrain sparsity theta = " << train_sparsity_theta->value() << ", "
              << "\n\tSpatsity phi = " << sparsity_phi->value() << ", "
              << "\n\tTest items processed = " << test_items_processed->value() << ", "
              << "\n\tTrain items processed = " << train_items_processed->value() << ", "
              << "\n\tKernel size = " << topic_kernel->average_kernel_size() << ", "
              << "\n\tKernel purity = " << topic_kernel->average_kernel_purity() << ", "
              << "\n\tKernel contrast = " << topic_kernel->average_kernel_contrast() << endl;
  }

  std::cout << endl;

  boost::timer::cpu_times elapsed = timer.elapsed();

  top_tokens = master_component.GetScoreAs<::artm::TopTokensScore>(model, "top_tokens");
  for (int topic_index = 0; topic_index < top_tokens.get()->values_size(); topic_index++) {
    std::cout << "#" << (topic_index+1) << ": ";
    auto top_tokens_for_topic = top_tokens.get()->values(topic_index);
    for (int token_index = 0; token_index < top_tokens_for_topic.value_size(); token_index++) {
      std::cout << top_tokens_for_topic.value(token_index) << " ";
    }
    std::cout << endl;
  }

  train_theta_snippet = master_component.GetScoreAs<::artm::ThetaSnippetScore>(model, "train_theta_snippet");
  int docs_to_show = train_theta_snippet.get()->values_size();
  std::cout << "\nThetaMatrix (first " << docs_to_show << " documents):\n";
  for (int topic_index = 0; topic_index < nTopics; topic_index++){
    std::cout << "Topic" << topic_index << ": ";
    for (int item_index = 0; item_index < docs_to_show; item_index++) {
      float weight = train_theta_snippet.get()->values(item_index).value(topic_index);
      std::cout << std::fixed << std::setw( 4 ) << std::setprecision( 5) << weight << " ";
    }
    std::cout << endl;
  }

  std::cout << "\nCPU TIME: " << (elapsed.user + elapsed.system) / 1e9 << " seconds"
            << "\nWALLCLOCK TIME: " << elapsed.wall / 1e9 << " seconds"
            << std::endl << std::endl;
}

int main(int argc, char * argv[]) {
  if (argc != 4) {
    cout << "Usage: cpp_client <docword> <vocab> nTopics" << endl;
    return 0;
  }

  int instance_size = 1;
  int processors_size = 2;
  try {
    proc(argc, argv, processors_size, instance_size);
  } catch (std::runtime_error& error) {
    cout << "Exception occured: " << error.what() << "\n";
  } catch (...) {
    cout << "Unknown exception occured.\n";
  }

  return 0;
}
