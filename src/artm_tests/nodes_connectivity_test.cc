// Copyright 2014, Additive Regularization of Topic Models.

#include "boost/thread.hpp"

#include "gtest/gtest.h"

#include "artm/core/node_controller.h"
#include "artm/core/master_component.h"
#include "artm/messages.pb.h"
#include "artm/core/instance.h"
#include "artm/core/data_loader.h"
#include "artm_tests/test_mother.h"
#include "artm/core/instance_schema.h"

// artm_tests.exe --gtest_filter=NodesConnectivityTest.*
TEST(NodesConnectivityTest, Basic) {
  ::artm::test::TestMother test_mother;

  ::artm::NodeControllerConfig node_config;
  node_config.set_create_endpoint("tcp://*:5556");
  int node_id = artm::core::NodeControllerManager::singleton().Create(node_config);
  auto node = artm::core::NodeControllerManager::singleton().Get(node_id);
  EXPECT_TRUE(node->impl()->instance() == nullptr);

  ::artm::MasterComponentConfig master_config;
  master_config.set_modus_operandi(::artm::MasterComponentConfig_ModusOperandi_Network);
  master_config.set_create_endpoint("tcp://*:5555");
  master_config.set_connect_endpoint("tcp://localhost:5555");
  master_config.add_node_connect_endpoint("tcp://localhost:5556");
  master_config.set_disk_path(".");

  auto& mcm = artm::core::MasterComponentManager::singleton();
  int master_id = mcm.Create<::artm::core::MasterComponent,
                             ::artm::MasterComponentConfig>(master_config);
  auto master = mcm.Get(master_id);
  EXPECT_FALSE(node->impl()->instance() == nullptr);

  auto regularizer_config = test_mother.GenerateRegularizerConfig();
  auto model_config = test_mother.GenerateModelConfig();
  master->CreateOrReconfigureRegularizer(regularizer_config);
  master->CreateOrReconfigureModel(model_config);
  auto schema = node->impl()->instance()->schema();
  EXPECT_TRUE(schema->has_model_config(model_config.name()));
  EXPECT_TRUE(schema->has_regularizer(regularizer_config.name()));
  master->DisposeModel(model_config.name());
  master->DisposeRegularizer(regularizer_config.name());

  master.reset();
  artm::core::MasterComponentManager::singleton().Erase(master_id);
  EXPECT_TRUE(node->impl()->instance() == nullptr);

  artm::core::NodeControllerManager::singleton().Erase(node_id);
  node.reset();
}
