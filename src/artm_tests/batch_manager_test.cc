// Copyright 2017, Additive Regularization of Topic Models.

#include <memory>

#include "gtest/gtest.h"

#include "artm/core/batch_manager.h"

#include "boost/uuid/random_generator.hpp"
#include "boost/uuid/uuid.hpp"

#include "artm/core/common.h"
#include "artm/core/thread_safe_holder.h"

// To run this particular test:
// artm_tests.exe --gtest_filter=BatchManager.*
TEST(BatchManager, Basic) {
  ::artm::core::BatchManager batch_manager;
  boost::uuids::random_generator new_uuid;

  boost::uuids::uuid u1(new_uuid()), u2(new_uuid()), u3(new_uuid());

  ASSERT_TRUE(batch_manager.IsEverythingProcessed());
  batch_manager.Add(u1);
  ASSERT_FALSE(batch_manager.IsEverythingProcessed());
  batch_manager.Callback(u1);
  ASSERT_TRUE(batch_manager.IsEverythingProcessed());

  batch_manager.Add(u2);
  batch_manager.Add(u3);

  ASSERT_FALSE(batch_manager.IsEverythingProcessed());

  batch_manager.Callback(u3);
  ASSERT_FALSE(batch_manager.IsEverythingProcessed());
  batch_manager.Callback(u2);
  ASSERT_TRUE(batch_manager.IsEverythingProcessed());
}
