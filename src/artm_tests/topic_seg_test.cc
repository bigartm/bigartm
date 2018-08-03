// Copyright 2017, Additive Regularization of Topic Models.

#include <memory>

#include "gtest/gtest.h"

#include "artm/cpp_interface.h"
#include "artm/core/common.h"
#include "artm/core/instance.h"

#include "artm_tests/test_mother.h"
#include "artm_tests/api.h"

// To run this particular test:
// artm_tests.exe --gtest_filter=Regularizer.TopicSegmentationPtdw
TEST(Regularizer, TopicSegmentationPtdw) {
  int n_topics = 5;
  int n_background_topics = 2;
  ::artm::MasterModelConfig master_config = ::artm::test::TestMother::GenerateMasterModelConfig(n_topics);
  master_config.set_cache_theta(true);
  master_config.set_num_document_passes(0);

  artm::MasterModel master_model_1(master_config);
  artm::MasterModel master_model_2(master_config);
  ::artm::test::Api api_1(master_model_1);
  ::artm::test::Api api_2(master_model_2);

  ::artm::Batch batch;
  batch.set_id(artm::test::Helpers::getUniqueString());
  batch.add_token("aaaa0");  // 0
  batch.add_token("bbbb1");  // 1
  batch.add_token("cccc2");  // 2
  batch.add_token("dddd3");  // 3
  batch.add_token("eeee4");  // 4
  batch.add_token("ffff5");  // 5
  artm::Item* item = batch.add_item();
  item->set_id(0);
  item->set_title("doc0");
  std::vector<int> token_sequence = {0, 1, 2, 0, 3, 2, 1, 4, 5};
  for (auto e : token_sequence) {
    item->add_token_id(e);
    item->add_transaction_start_index(item->transaction_start_index_size());
    item->add_token_weight(1.0);
  }
  item->add_transaction_start_index(item->transaction_start_index_size());

  std::vector<std::shared_ptr< ::artm::Batch>> batches;
  batches.push_back(std::make_shared< ::artm::Batch>(batch));
  auto offline_args_1 = api_1.Initialize(batches);
  auto offline_args_2 = api_2.Initialize(batches);

  for (int iter = 0; iter < 4; ++iter) {
    master_model_1.FitOfflineModel(offline_args_1);
    master_model_2.FitOfflineModel(offline_args_2);
  }

  ::artm::RegularizerConfig* regularizer_config = master_config.add_regularizer_config();
  regularizer_config->set_name("TopicSegmentationPtdwRegularizer");
  regularizer_config->set_type(::artm::RegularizerType_TopicSegmentationPtdw);
  regularizer_config->set_tau(0.0);
  ::artm::TopicSegmentationPtdwConfig internal_config;

  internal_config.set_window(3);
  internal_config.set_threshold(0.2);
  for (int i = 0; i < n_background_topics; ++i) {
    internal_config.add_background_topic_names("Topic" + boost::lexical_cast<std::string>(i));
  }
  regularizer_config->set_config(internal_config.SerializeAsString());

  master_model_1.Reconfigure(master_config);

  ::artm::TransformMasterModelArgs transform_args;
  transform_args.set_theta_matrix_type(::artm::ThetaMatrixType_DensePtdw);
  transform_args.add_batch_filename(batch.id());
  ::artm::ThetaMatrix ptdw_1 = master_model_1.Transform(transform_args);
  ::artm::ThetaMatrix ptdw_2 = master_model_2.Transform(transform_args);

  std::cout << "Ptdw_1 (reg):\n";
  for (int i = 0; i < ptdw_1.item_weights_size(); ++i) {
    std::cout << "token" << i << " profile: ";
    for (int k = 0; k < ptdw_1.item_weights(i).value_size(); ++k) {
      std::cout << ptdw_1.item_weights(i).value(k) << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "Ptdw_2 (no reg):\n";
  for (int i = 0; i < ptdw_2.item_weights_size(); ++i) {
    std::cout << "token" << i << " profile: ";
    for (int k = 0; k < ptdw_2.item_weights(i).value_size(); ++k) {
      std::cout << ptdw_2.item_weights(i).value(k) << ", ";
    }
    std::cout << "\n";
  }

  for (int i = 0; i < 7; ++i) {
    for (int k = 0; k < 5; ++k) {
      if (k == 0) {
        ASSERT_EQ(ptdw_1.item_weights(i).value(k), 1.0);
      } else {
        ASSERT_EQ(ptdw_1.item_weights(i).value(k), 0.0);
      }
    }
  }
  for (int i = 7; i < 9; ++i) {
    for (int k = 0; k < 5; ++k) {
      if (k < 4) {
        ASSERT_EQ(ptdw_1.item_weights(i).value(k), 0.0);
      } else {
        ASSERT_EQ(ptdw_1.item_weights(i).value(k), 1.0);
      }
    }
  }
}
