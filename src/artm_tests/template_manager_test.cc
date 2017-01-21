// Copyright 2017, Additive Regularization of Topic Models.

#include <memory>

#include "gtest/gtest.h"

#include "artm/core/common.h"
#include "artm/core/check_messages.h"
#include "artm/core/master_component.h"
#include "artm/core/template_manager.h"
#include "artm/core/helpers.h"

typedef artm::core::TemplateManager<std::shared_ptr< ::artm::core::MasterComponent>> MasterComponentManager;

// To run this particular test:
// artm_tests.exe --gtest_filter=TemplateManager.*
TEST(TemplateManager, Basic) {
  ::artm::MasterModelConfig config;
  ::artm::core::FixAndValidateMessage(&config, /* throw_error=*/ true);

  auto& mcm = MasterComponentManager::singleton();
  int id = mcm.Store(std::make_shared< ::artm::core::MasterComponent>(config));

  EXPECT_TRUE(mcm.Get(id) != nullptr);

  int id2 = mcm.Store(std::make_shared< ::artm::core::MasterComponent>(config));

  EXPECT_EQ(id2, id+1);
  EXPECT_TRUE(mcm.Get(id2) != nullptr);

  mcm.Erase(id);
  EXPECT_FALSE(mcm.Get(id) != nullptr);

  mcm.Erase(id2);
}
