// Copyright 2017, Additive Regularization of Topic Models.

#include <memory>

#include "gtest/gtest.h"

#include "boost/uuid/random_generator.hpp"
#include "boost/uuid/uuid.hpp"
#include "boost/filesystem.hpp"

#include "artm/cpp_interface.h"
#include "artm/core/common.h"
#include "artm_tests/test_mother.h"
#include "artm_tests/api.h"

void RunTest(bool disk_cache, std::string ptd_name) {
  const int nTokens = 10;
  const int batches_size = 3;
  const int nTopics = 8;

  std::string target_path = artm::test::Helpers::getUniqueString();

  artm::ConfigureLoggingArgs log_args;
  log_args.set_log_dir(".");
  ::artm::ConfigureLogging(log_args);
  EXPECT_EQ(log_args.log_dir(), ::google::GetLoggingDirectories()[0]);
  EXPECT_EQ(::google::GetLoggingDirectories().size(), 1);

  ::artm::MasterModelConfig master_config = ::artm::test::TestMother::GenerateMasterModelConfig(nTopics);
  master_config.set_reuse_theta(true);
  master_config.set_ptd_name(ptd_name);
  if (disk_cache) {
    master_config.set_disk_cache_path(target_path);
  }
  ::artm::MasterModel master_component(master_config);
  ::artm::test::Api api(master_component);
  EXPECT_TRUE(master_component.info().config().cache_theta());

  auto batches = ::artm::test::TestMother::GenerateBatches(batches_size, nTokens);
  auto fit_offline_args = api.Initialize(batches);

  for (int iter = 0; iter < 3; ++iter) {
    master_component.FitOfflineModel(fit_offline_args);
  }

  if (ptd_name.empty()) {
    EXPECT_GT(master_component.info().cache_entry_size(), 0);
  }
  ::artm::ThetaMatrix theta1 = master_component.GetThetaMatrix();
  EXPECT_EQ(theta1.num_topics(), nTopics);
  EXPECT_GE(theta1.item_id_size(), 1);
  auto config = master_component.config();
  config.set_num_document_passes(0);
  master_component.Reconfigure(config);
  master_component.FitOfflineModel(fit_offline_args);

  ::artm::ThetaMatrix theta2 = master_component.GetThetaMatrix();
  bool ok;
  ::artm::test::Helpers::CompareThetaMatrices(theta1, theta2, &ok);
  EXPECT_TRUE(ok);

  if (!ok) {
    ::artm::test::Helpers::DescribeThetaMatrix(theta1);
    ::artm::test::Helpers::DescribeThetaMatrix(theta2);
  }

  try { boost::filesystem::remove_all(target_path); }
  catch (...) { }
}

// To run this particular test:
// artm_tests.exe --gtest_filter=CacheManager.Basic
TEST(CacheManager, Basic) {
  RunTest(false, /*ptd_name=*/ "");
}

// To run this particular test:
// artm_tests.exe --gtest_filter=CacheManager.DiskCache
TEST(CacheManager, DiskCache) {
  RunTest(true, /*ptd_name=*/ "");
}

// To run this particular test:
// artm_tests.exe --gtest_filter=CacheManager.PtdName
TEST(CacheManager, PtdName) {
  RunTest(false, /*ptd_name=*/ "ptd");
}
