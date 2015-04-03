// Copyright 2014, Additive Regularization of Topic Models.

#include "boost/thread.hpp"
#include "gtest/gtest.h"

#include "boost/filesystem.hpp"

#include "artm/cpp_interface.h"
#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/messages.pb.h"

#include "artm/core/internals.pb.h"

#include "artm_tests/test_mother.h"

std::string runOfflineTest() {
  std::string target_path = artm::test::Helpers::getUniqueString();
  const int nTopics = 5;

  // Endpoints:
  // 5555 - master component (network_mode)
  // 5556 - node controller for workers (network_mode)
  // 5557 - node controller for master (proxy_mode)

  ::artm::MasterComponentConfig master_config;
  master_config.set_cache_theta(true);
  master_config.set_processors_count(1);
  ::artm::MasterComponent master_component(master_config);

  // Create model
  artm::ModelConfig model_config;
  model_config.set_topics_count(nTopics);
  model_config.set_name(artm::test::Helpers::getUniqueString());
  artm::Model model(master_component, model_config);

  std::vector<std::shared_ptr< ::artm::Batch>> batches;
  int batches_size = 2;
  int nTokens = 10;
  srand(1);
  for (int iBatch = 0; iBatch < batches_size; ++iBatch) {
    ::artm::Batch batch;
    batch.set_id(artm::test::Helpers::getUniqueString());

    // Same dictionary across all batches
    for (int i = 0; i < nTokens; i++) {
      std::stringstream str;
      str << "token" << i;
      batch.add_token(str.str());
    }

    artm::Item* item = batch.add_item();
    item->set_id(iBatch);  // one item per batch
    artm::Field* field = item->add_field();
    for (int iToken = 0; iToken < nTokens; ++iToken) {
      if (iToken == 0 || rand() % 3 == 0) {  // NOLINT
        field->add_token_id(iToken);
        field->add_token_count(1);
      }
    }

    batches.push_back(std::make_shared< ::artm::Batch>(batch));
  }

  for (int iter = 0; iter < 3; ++iter) {
    for (int iBatch = 0; iBatch < batches_size; ++iBatch) {
      master_component.AddBatch(*batches[iBatch]);
    }

    master_component.WaitIdle();
    model.Synchronize(0.0);
  }

  std::shared_ptr< ::artm::TopicModel> topic_model = master_component.GetTopicModel(model.name());
  std::stringstream ss;
  ss << "Topic model:\n";
  for (int i = 0; i < topic_model->token_size(); ++i) {
    ss << topic_model->token(i) << ": ";
    for (int j = 0; j < topic_model->topics_count(); ++j) {
      ss << topic_model->token_weights(i).value(j) << " ";
    }
    ss << std::endl;
  }

  ss << "Theta matrix:\n";
  for (int i = 0; i < batches_size; ++i) {
    ::artm::GetThetaMatrixArgs args;
    args.set_model_name(model.name());
    args.mutable_batch()->CopyFrom(*batches[i]);
    std::shared_ptr< ::artm::ThetaMatrix> theta_matrix = master_component.GetThetaMatrix(args);
    for (int i = 0; i < theta_matrix->item_id_size(); ++i) {
      ss << theta_matrix->item_id(i) << ": ";
      for (int j = 0; j < theta_matrix->topics_count(); ++j) {
        ss << theta_matrix->item_weights(i).value(j) << " ";
      }
      ss << std::endl;
    }
  }

  try { boost::filesystem::remove_all(target_path); }
  catch (...) {}

  return ss.str();
}

// artm_tests.exe --gtest_filter=RepeatableResult.Offline
TEST(RepeatableResult, Offline) {
  std::string first_result = runOfflineTest();
  std::string second_result = runOfflineTest();
  if (first_result != second_result)
    std::cout << first_result << "\n" << second_result;
  ASSERT_EQ(first_result, second_result);
}

// artm_tests.exe --gtest_filter=RepeatableResult.RandomGenerator
TEST(RepeatableResult, RandomGenerator) {
  int num = 10;
  size_t seed = 5;
  std::vector<float> first_result = ::artm::core::Helpers::GenerateRandomVector(num, seed);
  std::vector<float> second_result = ::artm::core::Helpers::GenerateRandomVector(num, seed);
  ASSERT_EQ(first_result.size(), num);
  ASSERT_EQ(second_result.size(), num);
  for (int i = 1; i < num; ++i) {
    ASSERT_EQ(first_result[i], second_result[i]);
    ASSERT_NE(first_result[i - 1], first_result[i]);
  }
}
