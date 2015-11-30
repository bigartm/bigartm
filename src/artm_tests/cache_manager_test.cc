// Copyright 2014, Additive Regularization of Topic Models.

#include <memory>

#include "gtest/gtest.h"

#include "boost/uuid/random_generator.hpp"
#include "boost/uuid/uuid.hpp"
#include "boost/filesystem.hpp"

#include "artm/messages.pb.h"
#include "artm/cpp_interface.h"
#include "artm_tests/test_mother.h"

void RunTest(bool disk_cache) {
  const int nTokens = 10;
  const int batches_size = 3;
  std::vector<std::shared_ptr< ::artm::Batch>> batches;
  ::artm::test::TestMother::GenerateBatches(batches_size, nTokens, &batches);

  std::string target_path = artm::test::Helpers::getUniqueString();

  ::artm::MasterComponentConfig master_config;
  master_config.set_cache_theta(true);
  if (disk_cache) master_config.set_disk_cache_path(target_path);
  ::artm::MasterComponent master_component(master_config);
  EXPECT_TRUE(master_component.info()->config().cache_theta());

  const int nTopics = 8;
  artm::ModelConfig model_config;
  model_config.set_topics_count(nTopics);
  model_config.set_reuse_theta(true);
  model_config.set_name(::artm::test::Helpers::getUniqueString());
  ::artm::Model model(master_component, model_config);

  for (int iter = 0; iter < 3; ++iter) {
    for (unsigned iBatch = 0; iBatch < batches.size(); ++iBatch)
      master_component.AddBatch(*batches[iBatch]);
    master_component.WaitIdle();
    model.Synchronize(0.0);
  }

  EXPECT_GT(master_component.info()->cache_entry_size(), 0);
  std::shared_ptr< ::artm::ThetaMatrix> theta1 = master_component.GetThetaMatrix(model.name());
  EXPECT_EQ(theta1->topics_count(), nTopics);
  EXPECT_GE(theta1->item_id_size(), 1);
  model_config.set_inner_iterations_count(0);
  model.Reconfigure(model_config);
  {
    for (unsigned iBatch = 0; iBatch < batches.size(); ++iBatch)
      master_component.AddBatch(*batches[iBatch]);
    master_component.WaitIdle();
    model.Synchronize(0.0);
  }

  std::shared_ptr< ::artm::ThetaMatrix> theta2 = master_component.GetThetaMatrix(model.name());
  bool ok;
  ::artm::test::Helpers::CompareThetaMatrices(*theta1, *theta2, &ok);
  EXPECT_TRUE(ok);

  if (!ok) {
    ::artm::test::Helpers::DescribeThetaMatrix(*theta1);
    ::artm::test::Helpers::DescribeThetaMatrix(*theta2);
  }

  try { boost::filesystem::remove_all(target_path); }
  catch (...) {}
}

// To run this particular test:
// artm_tests.exe --gtest_filter=CacheManager.Basic
TEST(CacheManager, Basic) {
  RunTest(false);
}

// To run this particular test:
// artm_tests.exe --gtest_filter=CacheManager.DiskCache
TEST(CacheManager, DiskCache) {
  RunTest(true);
}
