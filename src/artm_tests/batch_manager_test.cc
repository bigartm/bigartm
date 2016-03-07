// Copyright 2014, Additive Regularization of Topic Models.

#include "gtest/gtest.h"

#include "artm/core/batch_manager.h"

#include <memory>

#include "boost/uuid/random_generator.hpp"
#include "boost/uuid/uuid.hpp"

#include "artm/core/thread_safe_holder.h"
#include "artm/messages.pb.h"

// To run this particular test:
// artm_tests.exe --gtest_filter=BatchManager.*
TEST(BatchManager, Basic) {
  ::artm::core::BatchManager batch_manager;
  boost::uuids::random_generator new_uuid;

  std::string m1("model1"), m2("model2"), m3("model3");

  boost::uuids::uuid u1(new_uuid()), u2(new_uuid()), u3(new_uuid());

  ASSERT_TRUE(batch_manager.IsEverythingProcessed());
  batch_manager.Add(u1, "", m1);
  ASSERT_FALSE(batch_manager.IsEverythingProcessed());
  batch_manager.Add(u1, "", m2);
  batch_manager.Add(u1, "", m3);

  batch_manager.Callback(u1, m1);
  batch_manager.Callback(u1, m2);
  
  batch_manager.DisposeModel(m3);
  ASSERT_TRUE(batch_manager.IsEverythingProcessed());

  batch_manager.Add(u2, "", m1);
  batch_manager.Add(u2, "", m2);

  batch_manager.Callback(u2, m1);
  batch_manager.DisposeModel(m2);
  ASSERT_TRUE(batch_manager.IsEverythingProcessed());
}
