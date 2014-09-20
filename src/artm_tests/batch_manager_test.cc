// Copyright 2014, Additive Regularization of Topic Models.

#include "gtest/gtest.h"

#include "artm/core/batch_manager.h"

#include <memory>

#include "boost/uuid/random_generator.hpp"
#include "boost/uuid/uuid.hpp"

#include "artm/core/thread_safe_holder.h"
#include "artm/core/instance_schema.h"
#include "artm/messages.pb.h"

// To run this particular test:
// artm_tests.exe --gtest_filter=BatchManager.*
TEST(BatchManager, Basic) {
  ::artm::core::ThreadSafeHolder<::artm::core::InstanceSchema> schema_holder;
  ::artm::core::BatchManager batch_manager(&schema_holder);
  boost::uuids::random_generator new_uuid;

  auto schema = std::make_shared<::artm::core::InstanceSchema>();
  std::string m1("model1"), m2("model2"), m3("model3");
  schema->set_model_config(m1, std::make_shared<::artm::ModelConfig>());
  schema->set_model_config(m2, std::make_shared<::artm::ModelConfig>());
  schema->set_model_config(m3, std::make_shared<::artm::ModelConfig>());
  schema_holder.set(schema);

  boost::uuids::uuid u1(new_uuid()), u2(new_uuid()), u3(new_uuid());

  batch_manager.Add(u1);
  auto v1 = batch_manager.Next();
  ASSERT_EQ(v1, u1);
  ASSERT_FALSE(batch_manager.IsEverythingProcessed());

  batch_manager.Done(u1, m1);
  batch_manager.Done(u1, m2);
  
  batch_manager.DisposeModel(m3);
  schema->clear_model_config(m3);
  schema_holder.set(schema);
  ASSERT_TRUE(batch_manager.IsEverythingProcessed());

  batch_manager.Add(u2);
  auto v2 = batch_manager.Next();
  ASSERT_EQ(v2, u2);

  batch_manager.Add(u3);
  batch_manager.Add(u2);  // once again
  auto v3 = batch_manager.Next();
  ASSERT_EQ(v3, u3);

  auto v_nill = batch_manager.Next();
  ASSERT_TRUE(v_nill.is_nil());

  batch_manager.Done(u2, m1);
  batch_manager.DisposeModel(m2);
  schema->clear_model_config(m2);
  schema_holder.set(schema);

  auto v2_2 = batch_manager.Next();
  ASSERT_EQ(v2_2, u2);
  batch_manager.Done(v3, m1);
  batch_manager.Done(v2_2, m1);
  ASSERT_TRUE(batch_manager.IsEverythingProcessed());
}
