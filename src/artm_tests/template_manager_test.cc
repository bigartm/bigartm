// Copyright 2014, Additive Regularization of Topic Models.

#include "gtest/gtest.h"

#include "artm/messages.pb.h"
#include "artm/core/master_component.h"
#include "artm/core/template_manager.h"

using ::artm::core::MasterComponentManager;

// To run this particular test:
// artm_tests.exe --gtest_filter=TemplateManager.*
TEST(TemplateManager, Basic) {
  auto& mcm = MasterComponentManager::singleton();
  int id = mcm.Create<::artm::core::MasterComponent, ::artm::MasterComponentConfig>(
    ::artm::MasterComponentConfig());

  EXPECT_EQ(mcm.Get(id)->id(), id);

  int id2 = mcm.Create<::artm::core::MasterComponent, ::artm::MasterComponentConfig>(
    ::artm::MasterComponentConfig());

  EXPECT_EQ(id2, id+1);
  EXPECT_EQ(mcm.Get(id2)->id(), id2);

  bool succeeded = mcm.TryCreate<::artm::core::MasterComponent, ::artm::MasterComponentConfig>(
    id2, ::artm::MasterComponentConfig());
  EXPECT_FALSE(succeeded);

  mcm.Erase(id);
  EXPECT_FALSE(mcm.Get(id) != nullptr);

  mcm.Erase(id2);
}
