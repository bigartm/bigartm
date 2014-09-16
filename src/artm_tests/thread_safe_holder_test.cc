// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/thread_safe_holder.h"

#include <boost/thread/mutex.hpp>
#include <boost/utility.hpp>

#include "gtest/gtest.h"

using ::artm::core::ThreadSafeHolder;
using ::artm::core::ThreadSafeCollectionHolder;

// To run this particular test:
// artm_tests.exe --gtest_filter=ThreadSafeHolder.*
TEST(ThreadSafeHolder, Basic) {
  ThreadSafeHolder<double> int_holder;
  int_holder.set(std::make_shared<double>(5.0));
  EXPECT_EQ(*int_holder.get(), 5.0);

  ThreadSafeCollectionHolder<int, double> collection_holder;
  int key1 = 2, key2 = 3, key3 = 4;
  collection_holder.set(key1, std::make_shared<double>(7.0));
  collection_holder.set(key2, std::make_shared<double>(8.0));
  EXPECT_EQ(*collection_holder.get(key1), 7.0);
  EXPECT_EQ(*collection_holder.get(key2), 8.0);

  EXPECT_TRUE(collection_holder.has_key(key1));
  EXPECT_FALSE(collection_holder.has_key(key3));
  collection_holder.erase(key1);
  EXPECT_FALSE(collection_holder.has_key(key1));
}
