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
#include "artm/core/token.h"
#include "artm/core/transaction_type.h"

#include "artm/core/helpers.h"

#include "artm_tests/test_mother.h"
#include "artm_tests/api.h"

namespace fs = boost::filesystem;
namespace pb = google::protobuf;

// artm_tests.exe --gtest_filter=Transactions.BasicTest
TEST(Transactions, BasicTest) {
  const int nTopics = 3;
  const int nDocs = 8;
  const int nTokens = 8;

  std::string target_folder = artm::test::Helpers::getUniqueString();

  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_CollectionFormat_VowpalWabbit);
  config.set_target_folder(target_folder);
  config.set_docword_file_path((::artm::test::Helpers::getTestDataDir() /
    "vw_transaction_data_extended.txt").string());
  config.set_num_items_per_batch(10);

  ::artm::ParseCollection(config);

  ::artm::MasterModelConfig master_config;
  master_config.set_cache_theta(true);
  master_config.set_disk_cache_path(".");
  master_config.set_pwt_name("pwt");

  master_config.add_topic_name("topic_1");
  master_config.add_topic_name("topic_2");
  master_config.add_topic_name("topic_3");

  master_config.add_transaction_typename("@default_transaction");
  master_config.add_transaction_typename("trans1");
  master_config.add_transaction_typename("trans2");

  master_config.add_transaction_weight(1.0f);
  master_config.add_transaction_weight(1.0f);
  master_config.add_transaction_weight(1.0f);

  master_config.add_class_id("class_1");
  master_config.add_class_id("class_2");
  master_config.add_class_id("class_3");
  master_config.add_class_id("class_4");

  master_config.add_class_weight(1.0f);
  master_config.add_class_weight(1.0f);
  master_config.add_class_weight(1.0f);
  master_config.add_class_weight(1.0f);

  ::artm::ScoreConfig* score_config = master_config.add_score_config();
  score_config->set_config(::artm::PerplexityScoreConfig().SerializeAsString());
  score_config->set_type(::artm::ScoreType_Perplexity);
  score_config->set_name("PerplexityScore");

  artm::MasterModel master_model(master_config);
  ::artm::test::Api api(master_model);

  std::vector<std::shared_ptr<artm::Batch>> batches;
  fs::recursive_directory_iterator it(target_folder);
  fs::recursive_directory_iterator endit;
  while (it != endit) {
    if (fs::is_regular_file(*it) && it->path().extension() == ".batch") {
      artm::Batch batch;
      ::artm::core::Helpers::LoadMessage(it->path().string(), &batch);
      batches.push_back(std::make_shared<artm::Batch>(batch));
    }
    ++it;
  }

  ::artm::FitOfflineMasterModelArgs offline_args = api.Initialize(batches);

  // should be consistent with "../../../test_data/vw_transaction_data_extended.txt"
  std::unordered_map<int, std::vector<std::vector<artm::core::Token>>> doc_to_transactions;
  for (int d = 0; d < nDocs; ++d) {
    doc_to_transactions.emplace(std::make_pair(d, std::vector<std::vector<artm::core::Token>>()));
  }
  doc_to_transactions[0].push_back({ { "class_1", "token_1" } });
  doc_to_transactions[0].push_back({ { "class_1", "token_1" },
                                     { "class_2", "token_2" }
  });
  doc_to_transactions[1].push_back({ { "class_1", "token_2" } });
  doc_to_transactions[1].push_back({ { "class_1", "token_2" },
                                     { "class_2", "token_3" }
  });
  doc_to_transactions[2].push_back({ { "class_1", "token_3" } });
  doc_to_transactions[2].push_back({ { "class_1", "token_3" },
                                     { "class_2", "token_4" }
  });
  doc_to_transactions[3].push_back({ { "class_1", "token_1" } });
  doc_to_transactions[3].push_back({ { "class_1", "token_1" },
                                     { "class_2", "token_2" }
  });
  doc_to_transactions[4].push_back({ { "class_1", "token_2" } });
  doc_to_transactions[4].push_back({ { "class_1", "token_2" },
                                     { "class_2", "token_3" }
  });
  doc_to_transactions[5].push_back({ { "class_1", "token_3" } });
  doc_to_transactions[5].push_back({ { "class_1", "token_3" },
                                     { "class_2", "token_4" }
  });
  doc_to_transactions[6].push_back({ { "class_3", "token_5" } });
  doc_to_transactions[6].push_back({ { "class_4", "token_5" },
                                     { "class_2", "token_2" },
                                     { "class_1", "token_2" },
  });
  doc_to_transactions[7].push_back({ { "class_1", "token_1" },
                                     { "class_2", "token_2" }
  });
  doc_to_transactions[7].push_back({ { "class_1", "token_2" },
                                     { "class_2", "token_3" }
  });
  doc_to_transactions[7].push_back({ { "class_1", "token_1" } });

  int nIters = 5;
  std::unordered_map<artm::core::Token, int, artm::core::TokenHasher> token_to_index;
  for (int iter = 0; iter < nIters; ++iter) {
    master_model.FitOfflineModel(offline_args);

    ::artm::GetScoreValueArgs get_score_args;
    get_score_args.set_score_name("PerplexityScore");
    auto perplexity = master_model.GetScoreAs< ::artm::PerplexityScore>(get_score_args);
    std::cout << "Perplexity: " << perplexity.value() << std::endl;

    if (iter == nIters - 1) {
      artm::GetTopicModelArgs args;
      args.set_model_name(master_model.config().pwt_name());
      args.mutable_topic_name()->CopyFrom(master_model.config().topic_name());
      artm::TopicModel phi = master_model.GetTopicModel();

      ::artm::ThetaMatrix theta = master_model.GetThetaMatrix();

      ASSERT_EQ(phi.topic_name_size(), nTopics);
      ASSERT_EQ(phi.token_size(), nTokens);
      ASSERT_EQ(theta.topic_name_size(), nTopics);
      ASSERT_EQ(theta.item_id_size(), nDocs);

      token_to_index.clear();
      for (int i = 0; i < phi.token_size(); ++i) {
        token_to_index.emplace(
            artm::core::Token(phi.class_id(i), phi.token(i)), i);
      }

      for (int d = 0; d < nDocs; ++d) {
        const auto& transactions = doc_to_transactions[d];
        for (int x = 0; x < transactions.size(); ++x) {
          float p_xd = 0.0f;
          for (int t = 0; t < nTopics; ++t) {
            float val = theta.item_weights(d).value(t);
            for (const auto& transaction : transactions[x]) {
              val *= phi.token_weights(token_to_index[transaction]).value(t);
            }
            p_xd += val;
          }

          if (d == 0 || d == 3) {
            ASSERT_TRUE(std::abs(p_xd - 0.66f) < 0.01f);
          } else if (d == 1 || d == 2 || d == 4 || d == 5 || (d == 6 && x == 0)) {
            ASSERT_TRUE(std::abs(p_xd - 1.0f) < 0.01f);
          } else if ((d == 6 && x == 1) || (d == 7 && x == 1)) {
            ASSERT_TRUE(std::abs(p_xd - 0.33f) < 0.01f);
          } else if (d == 7) {
            ASSERT_TRUE(std::abs(p_xd - 0.44f) < 0.01f);
          } else {
            ASSERT_TRUE(false);
          }
        }
      }
    }
  }
}
