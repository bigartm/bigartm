// Copyright 2014, Additive Regularization of Topic Models.

#include "boost/filesystem.hpp"

#include "gtest/gtest.h"

#include "artm/messages.pb.h"
#include "artm/cpp_interface.h"

#include "artm_tests/test_mother.h"

namespace fs = boost::filesystem;

// To run this particular test:
// artm_tests.exe --gtest_filter=CollectionParser.*
TEST(CollectionParser, UciBagOfWords) {
  std::string target_folder = artm::test::Helpers::getUniqueString();

  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_Format_BagOfWordsUci);
  config.set_target_folder(target_folder);
  config.set_num_items_per_batch(1);
  config.set_vocab_file_path("../../../test_data/vocab.parser_test.txt");
  config.set_docword_file_path("../../../test_data/docword.parser_test.txt");

  ::artm::ParseCollection(config);

  boost::filesystem::recursive_directory_iterator it(target_folder);
  boost::filesystem::recursive_directory_iterator endit;
  int batches_count = 0;
  while (it != endit) {
    if (boost::filesystem::is_regular_file(*it) && it->path().extension() == ".batch") {
      batches_count++;
      std::shared_ptr<artm::Batch> batch = artm::LoadBatch(it->path().string());
      ASSERT_TRUE(batch->item_size() == 1 || batch->item_size() == 3);
      int tokens_size = batch->item(0).field(0).token_weight_size();
      ASSERT_TRUE(tokens_size == 2 || tokens_size == 3);
    }
    ++it;
  }

  ASSERT_EQ(batches_count, 2);

  artm::MasterComponentConfig master_config;
  ::artm::MasterComponent mc(master_config);
  artm::GatherDictionaryArgs gather_config;
  gather_config.set_data_path(target_folder);
  gather_config.set_vocab_file_path(config.vocab_file_path());
  gather_config.set_dictionary_target_name("mydictionary");
  mc.GatherDictionary(gather_config);

  auto dictionary = mc.GetDictionary("mydictionary");
  ASSERT_EQ(dictionary->token_size(), 3);

  EXPECT_EQ(dictionary->token(0), "token1");
  EXPECT_EQ(dictionary->token(1), "token2");
  EXPECT_EQ(dictionary->token(2), "token3");

  EXPECT_EQ(dictionary->class_id(0), "@default_class");
  EXPECT_EQ(dictionary->class_id(1), "@default_class");
  EXPECT_EQ(dictionary->class_id(2), "@default_class");

  EXPECT_EQ(dictionary->token_df(0), 1);
  EXPECT_EQ(dictionary->token_df(1), 2);
  EXPECT_EQ(dictionary->token_df(2), 2);

  EXPECT_EQ(dictionary->token_tf(0), 5);
  EXPECT_EQ(dictionary->token_tf(1), 4);
  EXPECT_EQ(dictionary->token_tf(2), 9);

  ASSERT_APPROX_EQ(dictionary->token_value(0), 5.0 / 18.0);
  ASSERT_APPROX_EQ(dictionary->token_value(1), 2.0 / 9.0);
  ASSERT_APPROX_EQ(dictionary->token_value(2), 0.5);

  try { boost::filesystem::remove_all(target_folder); }
  catch (...) {}
}

TEST(CollectionParser, ErrorHandling) {
  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_Format_BagOfWordsUci);

  config.set_vocab_file_path("../../../test_data/vocab.parser_test_non_unique.txt");
  config.set_docword_file_path("../../../test_data/docword.parser_test.txt");
  ASSERT_THROW(::artm::ParseCollection(config), artm::InvalidOperationException);

  config.set_vocab_file_path("../../../test_data/vocab.parser_test_empty_line.txt");
  config.set_docword_file_path("../../../test_data/docword.parser_test.txt");
  ASSERT_THROW(::artm::ParseCollection(config), artm::InvalidOperationException);

  config.set_vocab_file_path("no_such_file.txt");
  config.set_docword_file_path("../../../test_data/docword.parser_test.txt");
  ASSERT_THROW(::artm::ParseCollection(config), artm::DiskReadException);
}

TEST(CollectionParser, MatrixMarket) {
  std::string target_folder = artm::test::Helpers::getUniqueString();

  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_Format_MatrixMarket);
  config.set_target_folder(target_folder);
  config.set_num_items_per_batch(10000);
  config.set_vocab_file_path("../../../test_data/deerwestere.txt");
  config.set_docword_file_path("../../../test_data/deerwestere.mm");

  ::artm::ParseCollection(config);

  boost::filesystem::recursive_directory_iterator it(target_folder);
  boost::filesystem::recursive_directory_iterator endit;
  int batches_count = 0;
  while (it != endit) {
    if (boost::filesystem::is_regular_file(*it) && it->path().extension() == ".batch") {
      batches_count++;
      std::shared_ptr<artm::Batch> batch = artm::LoadBatch(it->path().string());
      ASSERT_EQ(batch->item_size(), 9);
    }
    ++it;
  }

  ASSERT_EQ(batches_count, 1);

  try { boost::filesystem::remove_all(target_folder); }
  catch (...) {}
}

TEST(CollectionParser, Multiclass) {
  std::string target_folder = artm::test::Helpers::getUniqueString();

  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_Format_BagOfWordsUci);
  config.set_target_folder(target_folder);
  config.set_vocab_file_path("../../../test_data/vocab.parser_test_multiclass.txt");
  config.set_docword_file_path("../../../test_data/docword.parser_test.txt");

  ::artm::ParseCollection(config);

  boost::filesystem::recursive_directory_iterator it(target_folder);
  boost::filesystem::recursive_directory_iterator endit;
  int batches_count = 0;
  while (it != endit) {
    if (boost::filesystem::is_regular_file(*it) && it->path().extension() == ".batch") {
      batches_count++;
      std::shared_ptr<artm::Batch> batch = artm::LoadBatch(it->path().string());
      ASSERT_EQ(batch->class_id_size(), 3);
      ASSERT_EQ(batch->class_id(0), "class1");
      ASSERT_EQ(batch->class_id(1), "class1");
      ASSERT_EQ(batch->class_id(2), "@default_class");
      ASSERT_EQ(batch->item_size(), 2);
    }
    ++it;
  }

  ASSERT_EQ(batches_count, 1);

  try { boost::filesystem::remove_all(target_folder); }
  catch (...) {}
}

// To run this particular test:
// artm_tests.exe --gtest_filter=CollectionParser.VowpalWabbit
TEST(CollectionParser, VowpalWabbit) {
  std::string target_folder = artm::test::Helpers::getUniqueString();

  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_Format_VowpalWabbit);
  config.set_target_folder(target_folder);
  config.set_docword_file_path("../../../test_data/vw_data.txt");
  config.set_num_items_per_batch(1);

  ::artm::ParseCollection(config);

  boost::filesystem::recursive_directory_iterator it(target_folder);
  boost::filesystem::recursive_directory_iterator endit;
  int batches_count = 0;
  while (it != endit) {
    if (boost::filesystem::is_regular_file(*it) && it->path().extension() == ".batch") {
      batches_count++;
      std::shared_ptr<artm::Batch> batch = artm::LoadBatch(it->path().string());
      ASSERT_TRUE(batch->class_id_size() == 3 || batch->class_id_size() == 2);
      for (int i = 0; i < batch->token_size(); ++i) {
        if (batch->token(i) == "hello" || batch->token(i) == "world")
          ASSERT_EQ(batch->class_id(i), "@default_class");
        if (batch->token(i) == "noname" || batch->token(i) == "alex")
          ASSERT_EQ(batch->class_id(i), "author");
      }
      ASSERT_EQ(batch->item_size(), 1);
    }
    ++it;
  }

  ASSERT_EQ(batches_count, 2);

  try { boost::filesystem::remove_all(target_folder); }
  catch (...) {}
}
