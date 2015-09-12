#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <fstream>

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
    std::cerr << message_ << " " << delta_ms.count() << " milliseconds.\n";
  }

 private:
  std::string message_;
  std::chrono::time_point<std::chrono::system_clock> start_;
};

std::vector<std::string> findFilesInDirectory(std::string root, std::string ext) {
  std::vector<std::string> retval;
  if (boost::filesystem::exists(root) && boost::filesystem::is_directory(root)) {
    boost::filesystem::recursive_directory_iterator it(root);
    boost::filesystem::recursive_directory_iterator endit;
    while (it != endit) {
      if (boost::filesystem::is_regular_file(*it) && it->path().extension() == ext) {
        retval.push_back(it->path().string());
      }
      ++it;
    }
  }
  return retval;
}

class ProgressScope {
 public:
   explicit ProgressScope(const std::string& message)  {
     std::cerr << message << "... ";
   }

   ~ProgressScope() {
     if (std::uncaught_exception()) {
       std::cerr << "Failed\n";
     } else {
       std::cerr << "OK.\n";
     }
   }
};

bool parseNumberOrPercent(std::string str, double* value, bool* fraction ) {
  if (str.empty())
    return false;

  bool percent = false;
  if (str[str.size() - 1] == '%') {
    percent = true;
    str = str.substr(0, str.size() - 1);
  }

  *value = 0;
  *fraction = true;
  try {
    *value = boost::lexical_cast<double>(str);
  }
  catch (...) {
    return false;
  }

  if (percent) {
    *value /= 100;
    return true;
  }

  if (*value >= 1.0) {
    *fraction = false;
  }

  return true;
}

struct artm_options {
  std::string docword;
  std::string vocab;
  std::string batch_folder;
  std::string disk_cache_folder;
  std::string dictionary_file;
  std::string load_model;
  std::string save_model;
  std::string write_model_readable;
  std::string write_predictions;
  std::string dictionary_min_df;
  std::string dictionary_max_df;
  int num_topics;
  int num_processors;
  int num_iters;
  int num_inner_iters;
  int items_per_batch;
  int update_every;
  int parsing_format;
  float tau0;
  float kappa;
  float tau_phi;
  float tau_theta;
  float tau_decor;
  bool b_paused;
  bool b_no_scores;
  bool b_reuse_theta;
  bool b_disable_avx_opt;
  bool b_use_dense_bow;
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
  std::cerr << "\nTop tokens for " << class_id << ":";
  int topic_index = -1;
  for (int i = 0; i < top_tokens.num_entries(); i++) {
    if (top_tokens.topic_index(i) != topic_index) {
      topic_index = top_tokens.topic_index(i);
      std::cerr << "\n#" << (topic_index + 1) << ": ";
    }

    std::cerr << top_tokens.token(i) << "(" << std::setw(2) << std::setprecision(2) << top_tokens.weight(i) << ") ";
  }
}

::artm::ProcessBatchesArgs ExtractProcessBatchesArgs(const ModelConfig& model_config) {
  ::artm::ProcessBatchesArgs args;
  args.set_inner_iterations_count(model_config.inner_iterations_count());
  args.set_stream_name(model_config.stream_name());
  if (model_config.has_opt_for_avx()) args.set_opt_for_avx(model_config.opt_for_avx());
  if (model_config.has_reuse_theta()) args.set_reuse_theta(model_config.reuse_theta());
  if (model_config.has_use_sparse_bow()) args.set_use_sparse_bow(model_config.use_sparse_bow());
  args.mutable_class_id()->CopyFrom(model_config.class_id());
  args.mutable_class_weight()->CopyFrom(model_config.class_weight());
  return args;
}

int execute(const artm_options& options) {
  bool online = (options.update_every > 0);

  const std::string dictionary_name = "dictionary";
  const std::string pwt_model_name = "pwt";
  const std::string nwt_model_name = "nwt";
  const std::string rwt_model_name = "rwt";
  const std::string nwt_hat_model_name = "nwt_hat";

  if (options.b_paused) {
    std::cerr << "Press any key to continue. ";
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
  if (options.b_reuse_theta) master_config.set_cache_theta(true);
  if (!options.disk_cache_folder.empty()) master_config.set_disk_cache_path(options.disk_cache_folder);

  ModelConfig model_config;
  model_config.set_topics_count(options.num_topics);
  model_config.set_inner_iterations_count(options.num_inner_iters);
  model_config.set_stream_name("train_stream");
  model_config.set_opt_for_avx(!options.b_disable_avx_opt);
  model_config.set_use_sparse_bow(!options.b_use_dense_bow);
  if (options.b_reuse_theta) model_config.set_reuse_theta(true);
  model_config.set_name("15081980-90a7-4767-ab85-7cb551c39339");  // randomly generated GUID
  if (options.class_id.size() > 0) {
    for (const std::string& class_id : options.class_id) {
      model_config.add_class_id(class_id);
      model_config.add_class_weight(1.0f);
    }
  }

  ProcessBatchesArgs process_batches_args = ExtractProcessBatchesArgs(model_config);
  RegularizeModelArgs regularize_model_args;
  NormalizeModelArgs normalize_model_args;

  configureStreams(&master_config);
  if (!options.b_no_scores)
    configureScores(&master_config, &model_config, options);

  // Step 2. Collection parsing
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

    ProgressScope scope("Parsing text collection");
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
    ::artm::ParseCollection(collection_parser_config);
  } else {
    if (!fs::exists(fs::path(working_batch_folder))) {
      std::cerr << "Unable to find batch folder: " << working_batch_folder;
      return 1;
    }

    int batch_files_count = findFilesInDirectory(working_batch_folder, ".batch").size();
    if (batch_files_count == 0) {
      std::cerr << "No batches found in " << working_batch_folder;
      return 1;
    }

    std::cerr << "Using " << batch_files_count << " batch found in folder '" << working_batch_folder << "'\n";
  }

  // Step 3. Create master component.
  std::shared_ptr<MasterComponent> master_component;
  master_component.reset(new MasterComponent(master_config));

  // Step 3.1. Import dictionary
  bool use_dictionary = false;
  std::string dictionary_full_filename = (fs::path(working_batch_folder) / options.dictionary_file).string();
  if (fs::exists(dictionary_full_filename)) {
    ProgressScope scope(std::string("Loading dictionary file from") + dictionary_full_filename);
    ImportDictionaryArgs import_dictionary_args;
    import_dictionary_args.set_file_name(dictionary_full_filename);
    import_dictionary_args.set_dictionary_name(dictionary_name);
    master_component->ImportDictionary(import_dictionary_args);
    use_dictionary = true;
  }
  else {
    std::cerr << "Dictionary file " << dictionary_full_filename << " does not exist; BigARTM will use all tokens from batches.\n";
  }

  // Step 4. Configure regularizers.
  std::vector<std::shared_ptr<artm::Regularizer>> regularizers;
  if (options.tau_theta != 0) {
    regularizers.push_back(std::make_shared<artm::Regularizer>(
      *master_component, configureThetaRegularizer(options.tau_theta, &model_config)));
    process_batches_args.add_regularizer_name(regularizers.back()->config().name());
    process_batches_args.add_regularizer_tau(options.tau_theta);
  }
  if (options.tau_phi != 0) {
    regularizers.push_back(std::make_shared<artm::Regularizer>(
      *master_component, configurePhiRegularizer(options.tau_phi, &model_config)));
    ::artm::RegularizerSettings* settings = regularize_model_args.add_regularizer_settings();
    settings->set_name(regularizers.back()->config().name());
    settings->set_tau(options.tau_phi);
    settings->set_use_relative_regularization(false);
  }
  if (options.tau_decor != 0) {
    regularizers.push_back(std::make_shared<artm::Regularizer>(
      *master_component, configureDecorRegularizer(options.tau_decor, &model_config)));
    ::artm::RegularizerSettings* settings = regularize_model_args.add_regularizer_settings();
    settings->set_name(regularizers.back()->config().name());
    settings->set_tau(options.tau_decor);
    settings->set_use_relative_regularization(false);
  }

  // Step 5. Create and initialize model.
  if (options.load_model.empty()) {
    InitializeModelArgs initialize_model_args;
    initialize_model_args.set_model_name(pwt_model_name);
    initialize_model_args.set_topics_count(model_config.topics_count());
    if (use_dictionary) {
      ProgressScope scope(std::string("Initializing random model from dictionary ") + options.dictionary_file);
      initialize_model_args.set_dictionary_name(dictionary_name);
      initialize_model_args.set_source_type(InitializeModelArgs_SourceType_Dictionary);
    }
    else {
      bool fraction;
      double value;
      if (parseNumberOrPercent(options.dictionary_min_df, &value, &fraction))  {
        ::artm::InitializeModelArgs_Filter* filter = initialize_model_args.add_filter();
        if (fraction) filter->set_min_percentage(value);
        else filter->set_min_items(value);
      } else {
        if (!options.dictionary_min_df.empty())
          std::cerr << "Error in parameter 'dictionary_min_df', the option will be ignored (" << options.dictionary_min_df << ")\n";
      }
      if (parseNumberOrPercent(options.dictionary_max_df, &value, &fraction))  {
        ::artm::InitializeModelArgs_Filter* filter = initialize_model_args.add_filter();
        if (fraction) filter->set_max_percentage(value);
        else filter->set_max_items(value);
      }
      else {
        if (!options.dictionary_max_df.empty())
          std::cerr << "Error in parameter 'dictionary_max_df', the option will be ignored (" << options.dictionary_max_df << ")\n";
      }

      ProgressScope scope(std::string("Initializing random model from batches in folder ") +
                          (options.batch_folder.empty() ? "<temp>" : working_batch_folder));
      initialize_model_args.set_disk_path(working_batch_folder);
      initialize_model_args.set_source_type(InitializeModelArgs_SourceType_Batches);
    }
    master_component->InitializeModel(initialize_model_args);
  }
  else {
    ProgressScope scope(std::string("Loading model from ") + options.load_model);
    ImportModelArgs import_model_args;
    import_model_args.set_model_name(pwt_model_name);
    import_model_args.set_file_name(options.load_model);
    master_component->ImportModel(import_model_args);
  }

  ::artm::GetTopicModelArgs get_model_args;
  get_model_args.set_request_type(::artm::GetTopicModelArgs_RequestType_Tokens);
  get_model_args.set_model_name(pwt_model_name);
  std::shared_ptr< ::artm::TopicModel> topic_model = master_component->GetTopicModel(get_model_args);
  std::cerr << "Number of tokens in the model: " << topic_model->token_size() << std::endl;

  std::vector<std::string> batch_file_names = findFilesInDirectory(working_batch_folder, ".batch");
  int update_count = 0;
  for (int iter = 0; iter < options.num_iters; ++iter) {
    {
      CuckooWatch timer("Iteration " + boost::lexical_cast<std::string>(iter + 1) + " took ");

      if (!online) {
        process_batches_args.set_pwt_source_name(pwt_model_name);
        process_batches_args.set_nwt_target_name(nwt_hat_model_name);
        for (auto& batch_filename : batch_file_names)
          process_batches_args.add_batch_filename(batch_filename);
        master_component->ProcessBatches(process_batches_args);
        process_batches_args.clear_batch_filename();

        if (regularize_model_args.regularizer_settings_size() > 0) {
          regularize_model_args.set_nwt_source_name(nwt_hat_model_name);
          regularize_model_args.set_pwt_source_name(pwt_model_name);
          regularize_model_args.set_rwt_target_name(rwt_model_name);
          master_component->RegularizeModel(regularize_model_args);
          normalize_model_args.set_rwt_source_name(rwt_model_name);
        }

        normalize_model_args.set_nwt_source_name(nwt_hat_model_name);
        normalize_model_args.set_pwt_target_name(pwt_model_name);
        master_component->NormalizeModel(normalize_model_args);
      } else {  // online
        for (int i = 0; i < batch_file_names.size(); ++i) {
          process_batches_args.set_reset_scores(i == 0);  // reset scores at the beginning of each iteration
          process_batches_args.add_batch_filename(batch_file_names[i]);
          int size = process_batches_args.batch_filename_size();
          if (size >= options.update_every || (i + 1) == batch_file_names.size()) {
            update_count++;
            process_batches_args.set_pwt_source_name(pwt_model_name);
            process_batches_args.set_nwt_target_name(nwt_hat_model_name);
            master_component->ProcessBatches(process_batches_args);

            double apply_weight = (update_count == 1) ? 1.0 : pow(options.tau0 + update_count, -options.kappa);
            double decay_weight = 1.0 - apply_weight;

            MergeModelArgs merge_model_args;
            merge_model_args.add_nwt_source_name(nwt_model_name);
            merge_model_args.add_source_weight(decay_weight);
            merge_model_args.add_nwt_source_name(nwt_hat_model_name);
            merge_model_args.add_source_weight(apply_weight);
            merge_model_args.set_nwt_target_name(nwt_model_name);
            master_component->MergeModel(merge_model_args);

            if (regularize_model_args.regularizer_settings_size() > 0) {
              regularize_model_args.set_nwt_source_name(nwt_model_name);
              regularize_model_args.set_pwt_source_name(pwt_model_name);
              regularize_model_args.set_rwt_target_name(rwt_model_name);
              master_component->RegularizeModel(regularize_model_args);
              normalize_model_args.set_rwt_source_name(rwt_model_name);
            }

            normalize_model_args.set_nwt_source_name(nwt_model_name);
            normalize_model_args.set_pwt_target_name(pwt_model_name);
            master_component->NormalizeModel(normalize_model_args);
            process_batches_args.clear_batch_filename();
          }
        }

      }
    }

    if (!options.b_no_scores) {
      auto test_perplexity = master_component->GetScoreAs< ::artm::PerplexityScore>(pwt_model_name, "test_perplexity");
      auto train_perplexity = master_component->GetScoreAs< ::artm::PerplexityScore>(pwt_model_name, "train_perplexity");
      auto test_sparsity_theta = master_component->GetScoreAs< ::artm::SparsityThetaScore>(pwt_model_name, "test_sparsity_theta");
      auto train_sparsity_theta = master_component->GetScoreAs< ::artm::SparsityThetaScore>(pwt_model_name, "train_sparsity_theta");
      auto sparsity_phi = master_component->GetScoreAs< ::artm::SparsityPhiScore>(pwt_model_name, "sparsity_phi");
      auto test_items_processed = master_component->GetScoreAs< ::artm::ItemsProcessedScore>(pwt_model_name, "test_items_processed");
      auto train_items_processed = master_component->GetScoreAs< ::artm::ItemsProcessedScore>(pwt_model_name, "train_items_processed");
      auto topic_kernel = master_component->GetScoreAs< ::artm::TopicKernelScore>(pwt_model_name, "topic_kernel");

      std::cerr
        <<   "\tTest perplexity = " << test_perplexity->value() << ", "
        << "\n\tTrain perplexity = " << train_perplexity->value() << ", "
        << "\n\tTest sparsity theta = " << test_sparsity_theta->value() << ", "
        << "\n\tTrain sparsity theta = " << train_sparsity_theta->value() << ", "
        << "\n\tSparsity phi = " << sparsity_phi->value() << ", "
        << "\n\tTest items processed = " << test_items_processed->value() << ", "
        << "\n\tTrain items processed = " << train_items_processed->value() << ", "
        << "\n\tKernel size = " << topic_kernel->average_kernel_size() << ", "
        << "\n\tKernel purity = " << topic_kernel->average_kernel_purity() << ", "
        << "\n\tKernel contrast = " << topic_kernel->average_kernel_contrast() << std::endl;
    }
  }

  if (!options.save_model.empty()) {
    ProgressScope scope(std::string("Saving model to ") + options.save_model);
    ExportModelArgs export_model_args;
    export_model_args.set_model_name(pwt_model_name);
    export_model_args.set_file_name(options.save_model);
    master_component->ExportModel(export_model_args);
  }

  if (!options.write_model_readable.empty()) {
    ProgressScope scope(std::string("Saving model in readable format to ") + options.write_model_readable);
    ::artm::Matrix matrix;
    std::shared_ptr< ::artm::TopicModel> model = master_component->GetTopicModel(pwt_model_name, &matrix);
    if (matrix.no_columns() != model->topics_count())
      throw "internal error (matrix.no_columns() != theta->topics_count())";

    std::ofstream output(options.write_model_readable);

    // header
    output << "token;class_id;";
    for (int j = 0; j < model->topics_count(); ++j) {
      if (model->topic_name_size() > 0)
        output << model->topic_name(j) << ";";
      else
        output << "topic" << j << ";";
    }
    output << std::endl;

    // bulk
    for (int i = 0; i < model->token_size(); ++i) {
      output << model->token(i) << ";";
      output << (model->class_id_size() == 0 ? "" : model->class_id(i)) << ";";
      for (int j = 0; j < model->topics_count(); ++j) {
        output << matrix(i, j) << ";";
      }
      output << std::endl;
    }
  }

  if (!options.write_predictions.empty()) {
    ProgressScope scope(std::string("Generating model predictions into ") + options.write_predictions);
    if (!master_config.cache_theta()) {
      master_config.set_cache_theta(true);
      master_component->Reconfigure(master_config);
    }

    process_batches_args.set_pwt_source_name(pwt_model_name);
    process_batches_args.clear_nwt_target_name();

    for (auto& batch_filename : batch_file_names)
      process_batches_args.add_batch_filename(batch_filename);
    master_component->ProcessBatches(process_batches_args);
    process_batches_args.clear_batch_filename();

    ::artm::Matrix matrix;
    std::shared_ptr< ::artm::ThetaMatrix> theta = master_component->GetThetaMatrix(pwt_model_name, &matrix);
    if (matrix.no_columns() != theta->topics_count())
      throw "internal error (matrix.no_columns() != theta->topics_count())";

    std::ofstream output(options.write_predictions);

    // header
    output << "id;title;";
    for (int j = 0; j < theta->topics_count(); ++j) {
      if (theta->topic_name_size() > 0)
        output << theta->topic_name(j) << ";";
      else
        output << "topic" << j << ";";
    }
    output << std::endl;

    std::vector<std::pair<int, int>> id_to_index;
    for (int i = 0; i < theta->item_id_size(); ++i)
      id_to_index.push_back(std::make_pair(theta->item_id(i), i));
    std::sort(id_to_index.begin(), id_to_index.end());

    // bulk
    for (int i = 0; i < theta->item_id_size(); ++i) {
      int index = id_to_index[i].second;
      output << theta->item_id(index) << ";";
      output << (theta->item_title_size() == 0 ? "" : theta->item_title(index)) << ";";
      for (int j = 0; j < theta->topics_count(); ++j) {
        output << matrix(index, j) << ";";
      }
      output << std::endl;
    }
  }

  if (!options.b_no_scores) {
    std::cerr << std::endl;

    if (options.class_id.empty()) {
      auto top_tokens = master_component->GetScoreAs< ::artm::TopTokensScore>(pwt_model_name, "top_tokens");
      showTopTokenScore(*top_tokens, "@default_class");
    } else {
      for (const std::string& class_id : options.class_id) {
        auto top_tokens = master_component->GetScoreAs< ::artm::TopTokensScore>(pwt_model_name, class_id + "_top_tokens");
        showTopTokenScore(*top_tokens, class_id);
      }
    }

    auto train_theta_snippet = master_component->GetScoreAs< ::artm::ThetaSnippetScore>(pwt_model_name, "train_theta_snippet");
    int docs_to_show = train_theta_snippet.get()->values_size();
    std::cerr << "\nThetaMatrix (last " << docs_to_show << " processed documents, ids = ";
    for (int item_index = 0; item_index < train_theta_snippet->item_id_size(); ++item_index) {
      if (item_index != 0) std::cerr << ",";
      std::cerr << train_theta_snippet->item_id(item_index);
    }
    std::cerr << "):\n";
    for (int topic_index = 0; topic_index < options.num_topics; topic_index++) {
      std::cerr << "Topic" << topic_index << ": ";
      for (int item_index = 0; item_index < docs_to_show; item_index++) {
        float weight = train_theta_snippet.get()->values(item_index).value(topic_index);
        std::cerr << std::fixed << std::setw(4) << std::setprecision(5) << weight << " ";
      }
      std::cerr << std::endl;
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
      ("num_processors,p", po::value(&options.num_processors)->default_value(0), "number of concurrent processors (default: auto-detect)")
      ("num_iters,i", po::value(&options.num_iters)->default_value(10), "number of outer iterations")
      ("load_model", po::value(&options.load_model)->default_value(""), "load model from file before processing")
      ("save_model", po::value(&options.save_model)->default_value(""), "save the model to binary file after processing")
      ("write_model_readable", po::value(&options.write_model_readable)->default_value(""), "output the model in a human-readable format")
      ("write_predictions", po::value(&options.write_predictions)->default_value(""), "write prediction in a human-readable format")
      ("dictionary_min_df", po::value(&options.dictionary_min_df)->default_value(""), "filter out tokens present in less than N documents / less than P% of documents")
      ("dictionary_max_df", po::value(&options.dictionary_max_df)->default_value(""), "filter out tokens present in less than N documents / less than P% of documents")
      ("num_inner_iters", po::value(&options.num_inner_iters)->default_value(10), "number of inner iterations")
      ("dictionary_file", po::value(&options.dictionary_file)->default_value("dictionary"), "filename of dictionary file")
      ("items_per_batch", po::value(&options.items_per_batch)->default_value(500), "number of items per batch")
      ("tau_phi", po::value(&options.tau_phi)->default_value(0.0f), "regularization coefficient for PHI matrix")
      ("tau_theta", po::value(&options.tau_theta)->default_value(0.0f), "regularization coefficient for THETA matrix")
      ("tau_decor", po::value(&options.tau_decor)->default_value(0.0f),
        "regularization coefficient for topics decorrelation "
        "(use with care, since this value heavily depends on the size of the dataset)")
      ("no_scores", po::bool_switch(&options.b_no_scores)->default_value(false), "disable calculation of all scores")
      ("update_every", po::value(&options.update_every)->default_value(0), "[online algorithm] requests an update of the model after update_every document")
      ("tau0", po::value(&options.tau0)->default_value(1024), "[online algorithm] weight option from online update formula")
      ("kappa", po::value(&options.kappa)->default_value(0.7f), "[online algorithm] exponent option from online update formula")
      ("parsing_format", po::value(&options.parsing_format)->default_value(0), "parsing format (0 - UCI, 1 - matrix market, 2 - vowpal wabbit)")
      ("class_id", po::value< std::vector<std::string> >(&options.class_id)->multitoken(), "class_id(s) for multiclass datasets")
    ;

    po::options_description experimental_options("Experimental options");
    experimental_options.add_options()
      ("paused", po::bool_switch(&options.b_paused)->default_value(false), "start paused and waits for a keystroke (allows to attach a debugger)")
      ("reuse_theta", po::bool_switch(&options.b_reuse_theta)->default_value(false), "reuse theta between iterations")
      ("disk_cache_folder", po::value(&options.disk_cache_folder)->default_value(""), "disk cache folder")
      ("disable_avx_opt", po::bool_switch(&options.b_disable_avx_opt)->default_value(false), "disable AVX optimization (gives similar behavior of the Processor component to BigARTM v0.5.4)")
      ("use_dense_bow", po::bool_switch(&options.b_use_dense_bow)->default_value(false), "use dense representation of bag-of-words data in processors")
    ;

    all_options.add(basic_options);
    all_options.add(experimental_options);

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

    if (show_help) {
      std::cerr << all_options;

      std::cerr << "\nExamples:\n";
      std::cerr << "\tcpp_client -d docword.kos.txt -v vocab.kos.txt\n";
      std::cerr << "\tset GLOG_logtostderr=1 & cpp_client -d docword.kos.txt -v vocab.kos.txt\n";
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
