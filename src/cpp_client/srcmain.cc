#include <stdlib.h>
#include <chrono>
#include <ctime>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#include <set>

#include "boost/lexical_cast.hpp"

#include "boost/uuid/uuid.hpp"
#include "boost/uuid/uuid_generators.hpp"
#include "boost/uuid/uuid_io.hpp"

#include "boost/filesystem.hpp"
namespace fs = boost::filesystem;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "artm/cpp_interface.h"
#include "artm/messages.pb.h"
#include "glog/logging.h"
using namespace artm;

class CuckooWatch {
 public:
  explicit CuckooWatch(std::string message)
    : message_(message), start_(std::chrono::system_clock::now()) {}
  ~CuckooWatch() {
    auto delta = (std::chrono::system_clock::now() - start_);
    auto delta_ms = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
    std::cout << message_ << " " << delta_ms.count() << " milliseconds.\n";
  }

 private:
  std::string message_;
  std::chrono::time_point<std::chrono::system_clock> start_;
};

int countFilesInDirectory(std::string root, std::string ext) {
  int retval = 0;
  if (boost::filesystem::exists(root) && boost::filesystem::is_directory(root)) {
    boost::filesystem::recursive_directory_iterator it(root);
    boost::filesystem::recursive_directory_iterator endit;
    while (it != endit) {
      if (boost::filesystem::is_regular_file(*it) && it->path().extension() == ext) {
        retval++;
      }
      ++it;
    }
  }
  return retval;
}

struct artm_options {
  std::string docword;
  std::string vocab;
  std::string batch_folder;
  std::string disk_cache_folder;
  std::string proxy;
  std::string localhost;
  std::string dictionary_file;
  int num_topics;
  int num_processors;
  int num_iters;
  int num_inner_iters;
  int items_per_batch;
  int communication_timeout;
  int port;
  int update_every;
  int parsing_format;
  int merger_queue_size;
  float tau_phi;
  float tau_theta;
  float tau_decor;
  bool b_paused;
  bool b_no_scores;
  bool b_reuse_theta;
  std::vector<std::string> nodes;
  std::vector<std::string> class_id;
};

void configureStreams(artm::MasterComponentConfig* master_config) {
  // Configure train and test streams
  Stream* train_stream = master_config->add_stream();
  Stream* test_stream  = master_config->add_stream();
  train_stream->set_name("train_stream");
  train_stream->set_type(Stream_Type_ItemIdModulus);
  train_stream->set_modulus(10);
  for (int i = 0; i <= 8; ++i) {
    train_stream->add_residuals(i);
  }

  test_stream->set_name("test_stream");
  test_stream->set_type(Stream_Type_ItemIdModulus);
  test_stream->set_modulus(10);
  test_stream->add_residuals(9);
}

void configureScores(artm::MasterComponentConfig* master_config, ModelConfig* model_config, const artm_options& options) {
  ::artm::ScoreConfig score_config;
  ::artm::PerplexityScoreConfig perplexity_config;
  perplexity_config.set_stream_name("test_stream");
  score_config.set_config(perplexity_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_Perplexity);
  score_config.set_name("test_perplexity");
  master_config->add_score_config()->CopyFrom(score_config);

  perplexity_config.set_stream_name("train_stream");
  score_config.set_config(perplexity_config.SerializeAsString());
  score_config.set_name("train_perplexity");
  master_config->add_score_config()->CopyFrom(score_config);

  ::artm::SparsityThetaScoreConfig sparsity_theta_config;
  sparsity_theta_config.set_stream_name("test_stream");
  score_config.set_config(sparsity_theta_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_SparsityTheta);
  score_config.set_name("test_sparsity_theta");
  master_config->add_score_config()->CopyFrom(score_config);

  sparsity_theta_config.set_stream_name("train_stream");
  score_config.set_config(sparsity_theta_config.SerializeAsString());
  score_config.set_name("train_sparsity_theta");
  master_config->add_score_config()->CopyFrom(score_config);

  ::artm::SparsityPhiScoreConfig sparsity_phi_config;
  score_config.set_config(sparsity_phi_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_SparsityPhi);
  score_config.set_name("sparsity_phi");
  master_config->add_score_config()->CopyFrom(score_config);

  ::artm::ItemsProcessedScoreConfig items_processed_config;
  items_processed_config.set_stream_name("test_stream");
  score_config.set_config(items_processed_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_ItemsProcessed);
  score_config.set_name("test_items_processed");
  master_config->add_score_config()->CopyFrom(score_config);

  items_processed_config.set_stream_name("train_stream");
  score_config.set_config(items_processed_config.SerializeAsString());
  score_config.set_name("train_items_processed");
  master_config->add_score_config()->CopyFrom(score_config);

  if (options.class_id.empty()) {
    ::artm::TopTokensScoreConfig top_tokens_config;
    top_tokens_config.set_num_tokens(6);
    score_config.set_config(top_tokens_config.SerializeAsString());
    score_config.set_type(::artm::ScoreConfig_Type_TopTokens);
    score_config.set_name("top_tokens");
    master_config->add_score_config()->CopyFrom(score_config);
  }
  else {
    for (const std::string& class_id : options.class_id) {
      ::artm::TopTokensScoreConfig top_tokens_config;
      top_tokens_config.set_num_tokens(6);
      top_tokens_config.set_class_id(class_id);
      score_config.set_config(top_tokens_config.SerializeAsString());
      score_config.set_type(::artm::ScoreConfig_Type_TopTokens);
      score_config.set_name(class_id + "_top_tokens");
      master_config->add_score_config()->CopyFrom(score_config);
    }
  }

  ::artm::ThetaSnippetScoreConfig theta_snippet_config;
  theta_snippet_config.set_stream_name("train_stream");
  theta_snippet_config.set_item_count(7);
  score_config.set_config(theta_snippet_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_ThetaSnippet);
  score_config.set_name("train_theta_snippet");
  master_config->add_score_config()->CopyFrom(score_config);

  ::artm::TopicKernelScoreConfig topic_kernel_config;
  std::string tr = topic_kernel_config.SerializeAsString();
  score_config.set_config(topic_kernel_config.SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_TopicKernel);
  score_config.set_name("topic_kernel");
  master_config->add_score_config()->CopyFrom(score_config);
}

artm::RegularizerConfig configurePhiRegularizer(float tau, ModelConfig* model_config) {
  RegularizerConfig regularizer_config;
  std::string name = "regularizer_smsp_phi";
  regularizer_config.set_name(name);
  regularizer_config.set_type(::artm::RegularizerConfig_Type_SmoothSparsePhi);
  regularizer_config.set_config(::artm::SmoothSparsePhiConfig().SerializeAsString());

  model_config->add_regularizer_name(name);
  model_config->add_regularizer_tau(tau);
  return regularizer_config;
}

artm::RegularizerConfig configureThetaRegularizer(float tau, ModelConfig* model_config) {
  RegularizerConfig regularizer_config;
  std::string name = "regularizer_smsp_theta";
  regularizer_config.set_name(name);
  regularizer_config.set_type(::artm::RegularizerConfig_Type_SmoothSparseTheta);
  regularizer_config.set_config(::artm::SmoothSparseThetaConfig().SerializeAsString());
  model_config->add_regularizer_name(name);
  model_config->add_regularizer_tau(tau);
  return regularizer_config;
}

artm::RegularizerConfig configureDecorRegularizer(float tau, ModelConfig* model_config) {
  RegularizerConfig regularizer_config;
  std::string name = "regularizer_decor_phi";
  regularizer_config.set_name(name);
  regularizer_config.set_type(::artm::RegularizerConfig_Type_DecorrelatorPhi);
  regularizer_config.set_config(::artm::DecorrelatorPhiConfig().SerializeAsString());
  model_config->add_regularizer_name(name);
  model_config->add_regularizer_tau(tau);
  return regularizer_config;
}

void configureItemsProcessedScore(artm::MasterComponentConfig* master_config, ModelConfig* model_config) {
  ::artm::ScoreConfig score_config;
  score_config.set_config(::artm::ItemsProcessedScoreConfig().SerializeAsString());
  score_config.set_type(::artm::ScoreConfig_Type_ItemsProcessed);
  score_config.set_name("items_processed");
  master_config->add_score_config()->CopyFrom(score_config);
}

void showTopTokenScore(const artm::TopTokensScore& top_tokens, std::string class_id) {
  std::cout << "\nTop tokens for " << class_id << ":";
  int topic_index = -1;
  for (int i = 0; i < top_tokens.num_entries(); i++) {
    if (top_tokens.topic_index(i) != topic_index) {
      topic_index = top_tokens.topic_index(i);
      std::cout << "\n#" << (topic_index + 1) << ": ";
    }

    std::cout << top_tokens.token(i) << "(" << std::setw(2) << std::setprecision(2) << top_tokens.weight(i) << ") ";
  }
}

int execute(const artm_options& options) {
  bool is_network_mode = (options.nodes.size() > 0);
  bool is_proxy = (!options.proxy.empty());
  bool online = (options.update_every > 0);

  if (options.b_paused) {
    std::cout << "Press any key to continue. ";
    getchar();
  }

  // There are options for data handling:
  // 1. User provides docword, vocab and batch_folder => cpp_client parses collection and stores it in batch_folder
  // 2. User provides docword, vocab, no batch_folder => cpp_client parses collection and stores it in temp folder
  // 3. User provides batch_folder, but no docword/vocab => cpp_client uses batches from batch_folder

  bool parse_collection = (!options.docword.empty());
  std::string working_batch_folder = options.batch_folder;
  if (options.batch_folder.empty())
    working_batch_folder = boost::lexical_cast<std::string>(boost::uuids::random_generator()());

  // Step 1. Configuration
  MasterComponentConfig master_config;
  master_config.set_disk_path(working_batch_folder);
  master_config.set_processors_count(options.num_processors);
  master_config.set_merger_queue_max_size(options.merger_queue_size);
  if (options.b_reuse_theta) master_config.set_cache_theta(true);
  if (!options.disk_cache_folder.empty()) master_config.set_disk_cache_path(options.disk_cache_folder);

  ModelConfig model_config;
  model_config.set_topics_count(options.num_topics);
  model_config.set_inner_iterations_count(options.num_inner_iters);
  model_config.set_stream_name("train_stream");
  if (options.b_reuse_theta) model_config.set_reuse_theta(true);
  model_config.set_name("15081980-90a7-4767-ab85-7cb551c39339");  // randomly generated GUID
  if (options.class_id.size() > 0) {
    for (const std::string& class_id : options.class_id) {
      model_config.add_class_id(class_id);
      model_config.add_class_weight(1.0f);
    }
  }

  configureStreams(&master_config);
  if (!options.b_no_scores)
    configureScores(&master_config, &model_config, options);

  configureItemsProcessedScore(&master_config, &model_config);

  if (is_network_mode) {
    master_config.set_modus_operandi(MasterComponentConfig_ModusOperandi_Network);
    master_config.set_create_endpoint("tcp://*:" + boost::lexical_cast<std::string>(options.port));
    master_config.set_connect_endpoint("tcp://" + options.localhost + ":" + boost::lexical_cast<std::string>(options.port));
    master_config.set_communication_timeout(options.communication_timeout);
    for (auto& node : options.nodes)
      master_config.add_node_connect_endpoint(node);
  } else {
    master_config.set_modus_operandi(MasterComponentConfig_ModusOperandi_Local);
  }

  // Step 2. Collection parsing

  std::shared_ptr<DictionaryConfig> unique_tokens;
  if (parse_collection) {
    if (fs::exists(fs::path(working_batch_folder)) && !fs::is_empty(fs::path(working_batch_folder))) {
      std::cerr << "Can not parse collection, target batch directory is not empty: " << working_batch_folder;
      return 1;
    }

    boost::system::error_code error;
    fs::create_directories(working_batch_folder, error);
    if (error) {
      std::cerr << "Unable to create batch folder: " << working_batch_folder;
      return 1;
    }

    std::cout << "Parsing text collection... ";
    ::artm::CollectionParserConfig collection_parser_config;
    if (options.parsing_format == 0) {
      collection_parser_config.set_format(CollectionParserConfig_Format_BagOfWordsUci);
    } else if (options.parsing_format == 1) {
      collection_parser_config.set_format(CollectionParserConfig_Format_MatrixMarket);
    } else if (options.parsing_format == 2) {
      collection_parser_config.set_format(CollectionParserConfig_Format_VowpalWabbit);
    } else {
      std::cerr << "Invalid parsing format options: " << options.parsing_format;
      return 1;
    }

    if (options.parsing_format != 2 && !options.docword.empty() && options.vocab.empty()) {
      std::cerr << "Error: no vocab file was specified. All formats except Vowpal Wabbit require both docword and vocab files.";
      return 1;
    }

    collection_parser_config.set_docword_file_path(options.docword);
    if (!options.vocab.empty())
      collection_parser_config.set_vocab_file_path(options.vocab);
    collection_parser_config.set_dictionary_file_name(options.dictionary_file);
    collection_parser_config.set_target_folder(working_batch_folder);
    collection_parser_config.set_num_items_per_batch(options.items_per_batch);
    unique_tokens = ::artm::ParseCollection(collection_parser_config);
    std::cout << "OK.\n";
  } else {
    if (!fs::exists(fs::path(working_batch_folder))) {
      std::cerr << "Unable to find batch folder: " << working_batch_folder;
      return 1;
    }

    int batch_files_count = countFilesInDirectory(working_batch_folder, ".batch");
    if (batch_files_count == 0) {
      std::cerr << "No batches found in " << working_batch_folder;
      return 1;
    }

    std::cout << "Using " << batch_files_count << " batch found in folder '" << working_batch_folder << "'\n";
    std::string dictionary_full_filename = (fs::path(working_batch_folder) / options.dictionary_file).string();
    if (fs::exists(dictionary_full_filename)) {
      std::cout << "Loading dictionary file... ";
      unique_tokens = ::artm::LoadDictionary(dictionary_full_filename);
      std::cout << "OK.\n";
    } else {
      std::cout << "Dictionary file " << dictionary_full_filename << " does not exist; BigARTM will use all tokens from batches.\n";
    }
  }

  // Step 3. Create master component.
  std::shared_ptr<MasterComponent> master_component;
  if (is_proxy) {
     MasterProxyConfig master_proxy_config;
     master_proxy_config.set_node_connect_endpoint(options.proxy);
     master_proxy_config.mutable_config()->CopyFrom(master_config);
     master_proxy_config.set_communication_timeout(options.communication_timeout);
     master_component.reset(new MasterComponent(master_proxy_config));
  } else {
    master_component.reset(new MasterComponent(master_config));
  }

  std::shared_ptr<Dictionary> dictionary;
  if (unique_tokens != nullptr)
    dictionary.reset(new Dictionary(*master_component, *unique_tokens));

  // Step 4. Configure regularizers.
  std::vector<std::shared_ptr<artm::Regularizer>> regularizers;
  if (options.tau_theta != 0)
    regularizers.push_back(std::make_shared<artm::Regularizer>(
      *master_component, configureThetaRegularizer(options.tau_theta, &model_config)));
  if (options.tau_phi != 0)
    regularizers.push_back(std::make_shared<artm::Regularizer>(
      *master_component, configurePhiRegularizer(options.tau_phi, &model_config)));
  if (options.tau_decor != 0)
    regularizers.push_back(std::make_shared<artm::Regularizer>(
      *master_component, configureDecorRegularizer(options.tau_decor, &model_config)));

  // Step 5. Create and initialize model.
  Model model(*master_component, model_config);
  if (dictionary != nullptr)
    model.Initialize(*dictionary);

  for (int iter = 0; iter < options.num_iters; ++iter) {
    {
      CuckooWatch timer("Iteration " + boost::lexical_cast<std::string>(iter + 1) + " took ");

      master_component->InvokeIteration(1);

      if (!online) {
        master_component->WaitIdle();
        model.Synchronize(0.0);
      } else {

        double kappa = 0.5;
        double tau0 = 64;

        bool done = false;
        bool first_sync = true;
        int next_items_processed = options.update_every;
        while (!done) {
          done = master_component->WaitIdle(10);  // wait 10 ms
          int current_items_processed = master_component->GetScoreAs< ::artm::ItemsProcessedScore>(model, "items_processed")->value();
          if (done || (current_items_processed >= next_items_processed)) {
            int update_count = current_items_processed / options.update_every;
            next_items_processed = current_items_processed + options.update_every;
            double rho = pow(tau0 + update_count, -kappa);
            double decay_weight = first_sync ? 0.0 : 1.0 - rho;
            model.Synchronize(decay_weight, rho, true);
            first_sync = false;
            std::cout << ".";
          }
        }

        std::cout << " ";
      }
    }

    if (!options.b_no_scores) {
      auto test_perplexity = master_component->GetScoreAs< ::artm::PerplexityScore>(model, "test_perplexity");
      auto train_perplexity = master_component->GetScoreAs< ::artm::PerplexityScore>(model, "train_perplexity");
      auto test_sparsity_theta = master_component->GetScoreAs< ::artm::SparsityThetaScore>(model, "test_sparsity_theta");
      auto train_sparsity_theta = master_component->GetScoreAs< ::artm::SparsityThetaScore>(model, "train_sparsity_theta");
      auto sparsity_phi = master_component->GetScoreAs< ::artm::SparsityPhiScore>(model, "sparsity_phi");
      auto test_items_processed = master_component->GetScoreAs< ::artm::ItemsProcessedScore>(model, "test_items_processed");
      auto train_items_processed = master_component->GetScoreAs< ::artm::ItemsProcessedScore>(model, "train_items_processed");
      auto topic_kernel = master_component->GetScoreAs< ::artm::TopicKernelScore>(model, "topic_kernel");

      std::cout
        <<   "\tTest perplexity = " << test_perplexity->value() << ", "
        << "\n\tTrain perplexity = " << train_perplexity->value() << ", "
        << "\n\tTest spatsity theta = " << test_sparsity_theta->value() << ", "
        << "\n\tTrain sparsity theta = " << train_sparsity_theta->value() << ", "
        << "\n\tSpatsity phi = " << sparsity_phi->value() << ", "
        << "\n\tTest items processed = " << test_items_processed->value() << ", "
        << "\n\tTrain items processed = " << train_items_processed->value() << ", "
        << "\n\tKernel size = " << topic_kernel->average_kernel_size() << ", "
        << "\n\tKernel purity = " << topic_kernel->average_kernel_purity() << ", "
        << "\n\tKernel contrast = " << topic_kernel->average_kernel_contrast() << std::endl;
    }
  }

  if (!options.b_no_scores) {
    std::cout << std::endl;

    if (options.class_id.empty()) {
      auto top_tokens = master_component->GetScoreAs< ::artm::TopTokensScore>(model, "top_tokens");
      showTopTokenScore(*top_tokens, "@default_class");
    } else {
      for (const std::string& class_id : options.class_id) {
        auto top_tokens = master_component->GetScoreAs< ::artm::TopTokensScore>(model, class_id + "_top_tokens");
        showTopTokenScore(*top_tokens, class_id);
      }
    }

    auto train_theta_snippet = master_component->GetScoreAs< ::artm::ThetaSnippetScore>(model, "train_theta_snippet");
    int docs_to_show = train_theta_snippet.get()->values_size();
    std::cout << "\nThetaMatrix (last " << docs_to_show << " processed documents, ids = ";
    for (int item_index = 0; item_index < train_theta_snippet->item_id_size(); ++item_index) {
      if (item_index != 0) std::cout << ",";
      std::cout << train_theta_snippet->item_id(item_index);
    }
    std::cout << "):\n";
    for (int topic_index = 0; topic_index < options.num_topics; topic_index++) {
      std::cout << "Topic" << topic_index << ": ";
      for (int item_index = 0; item_index < docs_to_show; item_index++) {
        float weight = train_theta_snippet.get()->values(item_index).value(topic_index);
        std::cout << std::fixed << std::setw(4) << std::setprecision(5) << weight << " ";
      }
      std::cout << std::endl;
    }
  }

  if (options.batch_folder.empty()) {
    try { boost::filesystem::remove_all(working_batch_folder); }
    catch (...) {}
  }

  return 0;
}

int main(int argc, char * argv[]) {
  try {
    artm_options options;

    po::options_description all_options("BigARTM - library for advanced topic modeling (http://bigartm.org)");

    po::options_description basic_options("Basic options");
    basic_options.add_options()
      ("help,h", "display this help message")
      ("docword,d", po::value(&options.docword), "docword file in UCI format")
      ("vocab,v", po::value(&options.vocab), "vocab file in UCI format")
      ("batch_folder,b", po::value(&options.batch_folder)->default_value(""),
        "If docword or vocab arguments are not provided, cpp_client will try to read pre-parsed batches from batch_folder location. "
        "Otherwise, if both docword and vocab arguments are provided, cpp_client will parse the data and store batches in batch_folder location. ")
      ("num_topic,t", po::value(&options.num_topics)->default_value(16), "number of topics")
      ("num_processors,p", po::value(&options.num_processors)->default_value(2), "number of concurrent processors")
      ("num_iters,i", po::value(&options.num_iters)->default_value(10), "number of outer iterations")
      ("num_inner_iters", po::value(&options.num_inner_iters)->default_value(10), "number of inner iterations")
      ("reuse_theta", po::bool_switch(&options.b_reuse_theta)->default_value(false), "reuse theta between iterations")
      ("dictionary_file", po::value(&options.dictionary_file)->default_value("dictionary"), "filename of dictionary file")
      ("items_per_batch", po::value(&options.items_per_batch)->default_value(500), "number of items per batch")
      ("tau_phi", po::value(&options.tau_phi)->default_value(0.0f), "regularization coefficient for PHI matrix")
      ("tau_theta", po::value(&options.tau_theta)->default_value(0.0f), "regularization coefficient for THETA matrix")
      ("tau_decor", po::value(&options.tau_decor)->default_value(0.0f),
        "regularization coefficient for topics decorrelation "
        "(use with care, since this value heavily depends on the size of the dataset)")
      ("paused", po::bool_switch(&options.b_paused)->default_value(false), "wait for keystroke (allows to attach a debugger)")
      ("no_scores", po::bool_switch(&options.b_no_scores)->default_value(false), "disable calculation of all scores")
      ("update_every", po::value(&options.update_every)->default_value(0), "[online algorithm] requests an update of the model after update_every document")
      ("parsing_format", po::value(&options.parsing_format)->default_value(0), "parsing format (0 - UCI, 1 - matrix market, 2 - vowpal wabbit)")
      ("disk_cache_folder", po::value(&options.disk_cache_folder)->default_value(""), "disk cache folder")
      ("merger_queue_size", po::value(&options.merger_queue_size), "size of the merger queue")
      ("class_id", po::value< std::vector<std::string> >(&options.class_id)->multitoken(), "class_id(s) for multiclass datasets")
    ;
    all_options.add(basic_options);

    po::options_description networking_options("Networking options");
    networking_options.add_options()
      ("nodes", po::value< std::vector<std::string> >(&options.nodes)->multitoken(), "endpoints of the remote nodes (enables network modus operandi)")
      ("localhost", po::value(&options.localhost)->default_value("localhost"), "DNS name or the IP address of the localhost")
      ("port", po::value(&options.port)->default_value(5550), "port to use for master node")
      ("proxy", po::value(&options.proxy)->default_value(""), "proxy endpoint")
      ("timeout", po::value(&options.communication_timeout)->default_value(1000), "network communication timeout in milliseconds")
    ;
    all_options.add(networking_options);

    po::variables_map vm;
    store(po::command_line_parser(argc, argv).options(all_options).run(), vm);
    notify(vm);

    // Uncomment next two lines to override commandline settings by code. DON'T COMMIT such change to git.
    // options.docword = "D:\\datasets\\docword.kos.txt";
    // options.vocab   = "D:\\datasets\\vocab.kos.txt";

    bool show_help = (vm.count("help") > 0);

    // Show help if user neither provided batch folder, nor docword/vocab files
    if (options.docword.empty() && options.batch_folder.empty())
      show_help = true;

    if (vm.count("merger_queue_size") == 0)
      options.merger_queue_size = options.num_processors;  // by default set queue size based on the number of processors

    if (show_help) {
      std::cout << all_options;

      std::cout << "\nExamples:\n";
      std::cout << "\tcpp_client -d docword.kos.txt -v vocab.kos.txt\n";
      std::cout << "\tset GLOG_logtostderr=1 & cpp_client -d docword.kos.txt -v vocab.kos.txt\n";
      return 1;
    }

    return execute(options);
  } catch (std::exception& e) {
    std::cerr << "Exception  : " << e.what() << "\n";
    return 1;
  }
  catch (...) {
    std::cerr << "Unknown error occurred.";
    return 1;
  }

  return 0;
}
