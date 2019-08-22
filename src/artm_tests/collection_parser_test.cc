// Copyright 2017, Additive Regularization of Topic Models.

#include "boost/filesystem.hpp"

#include "gtest/gtest.h"

#include "artm/cpp_interface.h"
#include "artm/core/common.h"
#include "artm/core/helpers.h"
#include "artm/core/token.h"

#include "artm_tests/test_mother.h"

namespace fs = boost::filesystem;

// To run this particular test:
// artm_tests.exe --gtest_filter=CollectionParser.*
TEST(CollectionParser, UciBagOfWords) {
  std::string target_folder = artm::test::Helpers::getUniqueString();

  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_CollectionFormat_BagOfWordsUci);
  config.set_target_folder(target_folder);
  config.set_num_items_per_batch(1);
  config.set_vocab_file_path((::artm::test::Helpers::getTestDataDir() / "vocab.parser_test.txt").string());
  config.set_docword_file_path((::artm::test::Helpers::getTestDataDir() / "docword.parser_test.txt").string());

  ::artm::ParseCollection(config);

  fs::recursive_directory_iterator it(target_folder);
  fs::recursive_directory_iterator endit;
  int batches_count = 0;
  while (it != endit) {
    if (fs::is_regular_file(*it) && it->path().extension() == ".batch") {
      batches_count++;
      ::artm::Batch batch;
      ::artm::core::Helpers::LoadMessage(it->path().string(), &batch);
      ASSERT_TRUE(batch.item_size() == 1 || batch.item_size() == 3);
      int tokens_size = batch.item(0).token_weight_size();
      ASSERT_TRUE(tokens_size == 2 || tokens_size == 3);
    }
    ++it;
  }

  ASSERT_EQ(batches_count, 2);

  artm::MasterModelConfig master_config;
  ::artm::MasterModel mc(master_config);

  auto dictionary_checker = [&mc, &target_folder] (
    const std::string &path_to_vocab,
    const std::string &dict_name) -> void {
    // first of all, we gather dictionary into the core
    artm::GatherDictionaryArgs gather_config;
    gather_config.set_data_path(target_folder);
    gather_config.set_vocab_file_path(path_to_vocab);
    gather_config.set_dictionary_target_name(dict_name);
    mc.GatherDictionary(gather_config);

    // next, we retrieve it from the core
    ::artm::GetDictionaryArgs get_dictionary_args;
    get_dictionary_args.set_dictionary_name(dict_name);
    auto dict = mc.GetDictionary(get_dictionary_args);

    // now we check its consistency
    ASSERT_EQ(dict.token_size(), 3);

    EXPECT_EQ(dict.token(0), "token1");
    EXPECT_EQ(dict.token(1), "token2");
    EXPECT_EQ(dict.token(2), "token3");

    EXPECT_EQ(dict.class_id(0), "@default_class");
    EXPECT_EQ(dict.class_id(1), "@default_class");
    EXPECT_EQ(dict.class_id(2), "@default_class");

    EXPECT_EQ(dict.token_df(0), 1);
    EXPECT_EQ(dict.token_df(1), 2);
    EXPECT_EQ(dict.token_df(2), 2);

    EXPECT_EQ(dict.token_tf(0), 5);
    EXPECT_EQ(dict.token_tf(1), 4);
    EXPECT_EQ(dict.token_tf(2), 9);

    ASSERT_APPROX_EQ(dict.token_value(0), 5.0 / 18.0);
    ASSERT_APPROX_EQ(dict.token_value(1), 2.0 / 9.0);
    ASSERT_APPROX_EQ(dict.token_value(2), 0.5);
  };

  dictionary_checker(config.vocab_file_path(), "default_dictionary");
  dictionary_checker((::artm::test::Helpers::getTestDataDir() / "vocab.parser_test_no_newline.txt").string(),
                     "no_newline_dictionary");

  try { fs::remove_all(target_folder); }
  catch (...) { }
}

TEST(CollectionParser, ErrorHandling) {
  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_CollectionFormat_BagOfWordsUci);

  config.set_vocab_file_path((::artm::test::Helpers::getTestDataDir() / "vocab.parser_test_non_unique.txt").string());
  config.set_docword_file_path((::artm::test::Helpers::getTestDataDir() / "docword.parser_test.txt").string());
  ASSERT_THROW(::artm::ParseCollection(config), artm::InvalidOperationException);

  config.set_vocab_file_path((::artm::test::Helpers::getTestDataDir() / "vocab.parser_test_empty_line.txt").string());
  config.set_docword_file_path((::artm::test::Helpers::getTestDataDir() / "docword.parser_test.txt").string());
  ASSERT_THROW(::artm::ParseCollection(config), artm::InvalidOperationException);

  config.set_vocab_file_path("no_such_file.txt");
  config.set_docword_file_path((::artm::test::Helpers::getTestDataDir() / "docword.parser_test.txt").string());
  ASSERT_THROW(::artm::ParseCollection(config), artm::DiskReadException);
}

TEST(CollectionParser, MatrixMarket) {
  std::string target_folder = artm::test::Helpers::getUniqueString();

  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_CollectionFormat_MatrixMarket);
  config.set_target_folder(target_folder);
  config.set_num_items_per_batch(10000);
  config.set_vocab_file_path((::artm::test::Helpers::getTestDataDir() / "deerwestere.txt").string());
  config.set_docword_file_path((::artm::test::Helpers::getTestDataDir() / "deerwestere.mm").string());

  ::artm::ParseCollection(config);

  fs::recursive_directory_iterator it(target_folder);
  fs::recursive_directory_iterator endit;
  int batches_count = 0;
  while (it != endit) {
    if (fs::is_regular_file(*it) && it->path().extension() == ".batch") {
      batches_count++;

      artm::Batch batch;
      ::artm::core::Helpers::LoadMessage(it->path().string(), &batch);
      ASSERT_EQ(batch.item_size(), 9);
    }
    ++it;
  }

  ASSERT_EQ(batches_count, 1);

  try { fs::remove_all(target_folder); }
  catch (...) { }
}

TEST(CollectionParser, Multiclass) {
  std::string target_folder = artm::test::Helpers::getUniqueString();

  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_CollectionFormat_BagOfWordsUci);
  config.set_target_folder(target_folder);
  config.set_vocab_file_path((::artm::test::Helpers::getTestDataDir() / "vocab.parser_test_multiclass.txt").string());
  config.set_docword_file_path((::artm::test::Helpers::getTestDataDir() / "docword.parser_test.txt").string());

  ::artm::ParseCollection(config);

  fs::recursive_directory_iterator it(target_folder);
  fs::recursive_directory_iterator endit;
  int batches_count = 0;
  while (it != endit) {
    if (fs::is_regular_file(*it) && it->path().extension() == ".batch") {
      batches_count++;
      artm::Batch batch;
      ::artm::core::Helpers::LoadMessage(it->path().string(), &batch);
      ASSERT_EQ(batch.class_id_size(), 3);
      ASSERT_EQ(batch.class_id(0), "class1");
      ASSERT_EQ(batch.class_id(1), "class1");
      ASSERT_EQ(batch.class_id(2), "@default_class");
      ASSERT_EQ(batch.item_size(), 2);
    }
    ++it;
  }

  ASSERT_EQ(batches_count, 1);

  std::string dictionary_name = "dictionary";
  artm::GatherDictionaryArgs gather_args;
  gather_args.set_data_path(target_folder);
  gather_args.set_dictionary_target_name(dictionary_name);
  gather_args.set_vocab_file_path((::artm::test::Helpers::getTestDataDir() /
                                   "vocab.parser_test_multiclass.txt").string());

  ::artm::MasterModelConfig master_config;
  artm::MasterModel master(master_config);
  master.GatherDictionary(gather_args);
  ::artm::GetDictionaryArgs get_dictionary_args;
  get_dictionary_args.set_dictionary_name(dictionary_name);
  auto dictionary_ptr = master.GetDictionary(get_dictionary_args);

  ASSERT_EQ(dictionary_ptr.token_size(), 3);
  ASSERT_EQ(dictionary_ptr.class_id_size(), 3);
  ASSERT_EQ(dictionary_ptr.token_tf_size(), 3);
  ASSERT_EQ(dictionary_ptr.token_df_size(), 3);
  ASSERT_EQ(dictionary_ptr.token_value_size(), 3);

  ASSERT_EQ(dictionary_ptr.token(0), "token1");
  ASSERT_EQ(dictionary_ptr.token(1), "token2");
  ASSERT_EQ(dictionary_ptr.token(2), "token3");

  ASSERT_EQ(dictionary_ptr.class_id(0), "class1");
  ASSERT_EQ(dictionary_ptr.class_id(1), artm::core::DefaultClass);
  ASSERT_EQ(dictionary_ptr.class_id(2), "class1");

  ASSERT_APPROX_EQ(dictionary_ptr.token_df(0), 1);
  ASSERT_APPROX_EQ(dictionary_ptr.token_df(1), 2);
  ASSERT_APPROX_EQ(dictionary_ptr.token_df(2), 2);

  ASSERT_APPROX_EQ(dictionary_ptr.token_tf(0), 5.0);
  ASSERT_APPROX_EQ(dictionary_ptr.token_tf(1), 4.0);
  ASSERT_APPROX_EQ(dictionary_ptr.token_tf(2), 9.0);

  ASSERT_APPROX_EQ(dictionary_ptr.token_value(0), 5.0 / 14.0);
  ASSERT_APPROX_EQ(dictionary_ptr.token_value(1), 4.0 / 4.0);
  ASSERT_APPROX_EQ(dictionary_ptr.token_value(2), 9.0 / 14.0);

  try { fs::remove_all(target_folder); }
  catch (...) { }
}

// To run this particular test:
// artm_tests.exe --gtest_filter=CollectionParser.VowpalWabbit
TEST(CollectionParser, VowpalWabbit) {
  std::string target_folder = artm::test::Helpers::getUniqueString();

  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_CollectionFormat_VowpalWabbit);
  config.set_target_folder(target_folder);
  config.set_docword_file_path((::artm::test::Helpers::getTestDataDir() / "vw_data.txt").string());
  config.set_num_items_per_batch(1);

  ::artm::ParseCollection(config);

  fs::recursive_directory_iterator it(target_folder);
  fs::recursive_directory_iterator endit;
  int batches_count = 0;
  while (it != endit) {
    if (fs::is_regular_file(*it) && it->path().extension() == ".batch") {
      batches_count++;
      ::artm::Batch batch;
      ::artm::core::Helpers::LoadMessage(it->path().string(), &batch);
      ASSERT_TRUE(batch.class_id_size() == 3 || batch.class_id_size() == 2);
      for (int i = 0; i < batch.token_size(); ++i) {
        if (batch.token(i) == "hello" || batch.token(i) == "world") {
          ASSERT_EQ(batch.class_id(i), "@default_class");
        }
        if (batch.token(i) == "noname" || batch.token(i) == "alex") {
          ASSERT_EQ(batch.class_id(i), "author");
        }
      }
      ASSERT_EQ(batch.item_size(), 1);
    }
    ++it;
  }

  ASSERT_EQ(batches_count, 2);

  try { fs::remove_all(target_folder); }
  catch (...) {}
}

// To run this particular test:
// artm_tests.exe --gtest_filter=CollectionParser.TransactionVowpalWabbit
TEST(CollectionParser, TransactionVowpalWabbit) {
  std::string target_folder = artm::test::Helpers::getUniqueString();

  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_CollectionFormat_VowpalWabbit);
  config.set_target_folder(target_folder);
  config.set_docword_file_path((::artm::test::Helpers::getTestDataDir() / "vw_transaction_data.txt").string());
  config.set_num_items_per_batch(2);

  ::artm::ParseCollection(config);

  fs::recursive_directory_iterator it(target_folder);
  fs::recursive_directory_iterator endit;
  int batches_count = 0;
  while (it != endit) {
    if (fs::is_regular_file(*it) && it->path().extension() == ".batch") {
      batches_count++;
      ::artm::Batch batch;
      ::artm::core::Helpers::LoadMessage(it->path().string(), &batch);

      ASSERT_EQ(batch.class_id_size(), batch.token_size());
      ASSERT_EQ(batch.class_id_size(), 8);

      for (int i = 0; i < batch.token_size(); ++i) {
        if (batch.token(i) == "hello" || batch.token(i) == "world") {
          ASSERT_EQ(batch.class_id(i), "@default_class");
        } else if (batch.token(i) == "click" || batch.token(i) == "show") {
          ASSERT_EQ(batch.class_id(i), "action");
        } else if (batch.token(i) == "twice" || batch.token(i) == "first") {
          ASSERT_EQ(batch.class_id(i), "qualifier");
        } else if (batch.token(i) == "mel-lain") {
          ASSERT_TRUE(batch.class_id(i) == "user" || batch.class_id(i) == "author");
        } else {
          ASSERT_EQ(batch.token(i), "hello");  // we should not get here
        }
      }

      ASSERT_EQ(batch.item_size(), 2);

      ASSERT_EQ(batch.item(0).token_id_size(), 6);
      ASSERT_EQ(batch.item(0).transaction_start_index_size(), 5);
      ASSERT_EQ(batch.item(0).token_weight_size(), 6);
      ASSERT_EQ(batch.item(1).token_id_size(), 8);
      ASSERT_EQ(batch.item(1).transaction_start_index_size(), 5);
      ASSERT_EQ(batch.item(1).token_weight_size(), 8);

      // check first item
      ASSERT_FLOAT_EQ(batch.item(0).token_weight(0), 1.0);
      ASSERT_FLOAT_EQ(batch.item(0).token_weight(1), 2.0);
      ASSERT_FLOAT_EQ(batch.item(0).token_weight(2), 3.0);
      ASSERT_FLOAT_EQ(batch.item(0).token_weight(3), 3.0);
      ASSERT_FLOAT_EQ(batch.item(0).token_weight(4), 1.0);
      ASSERT_FLOAT_EQ(batch.item(0).token_weight(5), 1.0);


      ASSERT_EQ(batch.item(0).transaction_start_index(0), 0);
      ASSERT_EQ(batch.item(0).transaction_start_index(1), 1);
      ASSERT_EQ(batch.item(0).transaction_start_index(2), 2);
      ASSERT_EQ(batch.item(0).transaction_start_index(3), 4);
      ASSERT_EQ(batch.item(0).transaction_start_index(4), 6);

      // both are ids of "mel-lain" as "user"
      ASSERT_EQ(batch.item(0).token_id(2),
                batch.item(0).token_id(4));

      // check second item
      ASSERT_FLOAT_EQ(batch.item(1).token_weight(0), 1.0);
      ASSERT_FLOAT_EQ(batch.item(1).token_weight(1), 5.0);
      ASSERT_FLOAT_EQ(batch.item(1).token_weight(2), 5.0);
      ASSERT_FLOAT_EQ(batch.item(1).token_weight(3), 5.0);
      ASSERT_FLOAT_EQ(batch.item(1).token_weight(4), 1.0);
      ASSERT_FLOAT_EQ(batch.item(1).token_weight(5), 1.0);
      ASSERT_FLOAT_EQ(batch.item(1).token_weight(6), 1.0);
      ASSERT_FLOAT_EQ(batch.item(1).token_weight(7), 1.0);

      ASSERT_EQ(batch.item(1).transaction_start_index(0), 0);
      ASSERT_EQ(batch.item(1).transaction_start_index(1), 1);
      ASSERT_EQ(batch.item(1).transaction_start_index(2), 4);
      ASSERT_EQ(batch.item(1).transaction_start_index(3), 7);
      ASSERT_EQ(batch.item(1).transaction_start_index(4), 8);

      // both are ids of "wordl" as "@default_class"
      ASSERT_EQ(batch.item(1).token_id(4),
                batch.item(1).token_id(7));
    }
    ++it;
  }
  ASSERT_EQ(batches_count, 1);

  try { fs::remove_all(target_folder); }
  catch (...) {}
}
// vim: set ts=2 sw=2 sts=2:
