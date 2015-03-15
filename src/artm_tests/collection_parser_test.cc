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
  config.set_dictionary_file_name("test_parser.dictionary");
  config.set_cooccurrence_file_name("test_parser.cooc.dictionary");
  config.add_cooccurrence_token("token1");
  config.add_cooccurrence_token("token2");
  config.add_cooccurrence_token("token3");
  config.set_num_items_per_batch(1);
  config.set_vocab_file_path("../../../test_data/vocab.parser_test.txt");
  config.set_docword_file_path("../../../test_data/docword.parser_test.txt");

  std::shared_ptr< ::artm::DictionaryConfig> dictionary_parsed = ::artm::ParseCollection(config);
  ASSERT_EQ(dictionary_parsed->entry_size(), 3);

  std::shared_ptr< ::artm::DictionaryConfig> dictionary_loaded = ::artm::LoadDictionary(
    (fs::path(target_folder) / "test_parser.dictionary").string());
  ASSERT_EQ(dictionary_parsed->entry_size(), dictionary_loaded->entry_size());

  ASSERT_EQ(dictionary_loaded->entry_size(), 3);
  ASSERT_EQ(dictionary_loaded->total_token_count(), 18);
  ASSERT_EQ(dictionary_loaded->total_items_count(), 2);
  ASSERT_EQ(dictionary_loaded->entry(0).key_token(), "token1");
  ASSERT_EQ(dictionary_loaded->entry(0).class_id(), "@default_class");
  ASSERT_EQ(dictionary_loaded->entry(0).items_count(), 1);
  ASSERT_EQ(dictionary_loaded->entry(0).token_count(), 5);
  ASSERT_GT(dictionary_loaded->entry(0).value(), 0);
  ASSERT_EQ(dictionary_loaded->entry(1).key_token(), "token2");
  ASSERT_EQ(dictionary_loaded->entry(1).class_id(), "@default_class");
  ASSERT_EQ(dictionary_loaded->entry(1).items_count(), 2);
  ASSERT_EQ(dictionary_loaded->entry(1).token_count(), 4);
  ASSERT_EQ(dictionary_loaded->entry(2).key_token(), "token3");
  ASSERT_EQ(dictionary_loaded->entry(2).class_id(), "@default_class");
  ASSERT_EQ(dictionary_loaded->entry(2).items_count(), 2);
  ASSERT_EQ(dictionary_loaded->entry(2).token_count(), 9);

  std::shared_ptr< ::artm::DictionaryConfig> cooc_dictionary_loaded = ::artm::LoadDictionary(
    (fs::path(target_folder) / fs::path("test_parser.cooc.dictionary")).string());
  ASSERT_EQ(cooc_dictionary_loaded->entry_size(), 3);
  ASSERT_EQ(cooc_dictionary_loaded->entry(0).key_token(), "token1~token2");
  ASSERT_EQ(cooc_dictionary_loaded->entry(0).items_count(), 1);
  ASSERT_EQ(cooc_dictionary_loaded->entry(1).key_token(), "token1~token3");
  ASSERT_EQ(cooc_dictionary_loaded->entry(1).items_count(), 1);
  ASSERT_EQ(cooc_dictionary_loaded->entry(2).key_token(), "token2~token3");
  ASSERT_EQ(cooc_dictionary_loaded->entry(2).items_count(), 2);

  boost::filesystem::recursive_directory_iterator it(target_folder);
  boost::filesystem::recursive_directory_iterator endit;
  int batches_count = 0;
  while (it != endit) {
    if (boost::filesystem::is_regular_file(*it) && it->path().extension() == ".batch") {
      batches_count++;
      std::shared_ptr<artm::Batch> batch = artm::LoadBatch(it->path().string());
      ASSERT_TRUE(batch->item_size() == 1 || batch->item_size() == 3);
      int tokens_size = batch->item(0).field(0).token_count_size();
      ASSERT_TRUE(tokens_size == 2 || tokens_size == 3);
    }
    ++it;
  }

  ASSERT_EQ(batches_count, 2);

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

  config.set_vocab_file_path("../../../test_data/vocab.parser_test.txt");
  config.set_docword_file_path("no_such_file");
  ASSERT_THROW(::artm::ParseCollection(config), artm::DiskReadException);

  ASSERT_THROW(::artm::LoadDictionary("no_such_file"), artm::DiskReadException);
}

TEST(CollectionParser, MatrixMarket) {
  std::string target_folder = artm::test::Helpers::getUniqueString();

  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_Format_MatrixMarket);
  config.set_target_folder(target_folder);
  config.set_num_items_per_batch(10000);
  config.set_vocab_file_path("../../../test_data/deerwestere.txt");
  config.set_docword_file_path("../../../test_data/deerwestere.mm");
  config.set_dictionary_file_name("test_parser.dictionary");

  std::shared_ptr< ::artm::DictionaryConfig> dictionary_parsed = ::artm::ParseCollection(config);
  ASSERT_EQ(dictionary_parsed->entry_size(), 12);

  try { boost::filesystem::remove_all(target_folder); }
  catch (...) {}
}

TEST(CollectionParser, Multiclass) {
  std::string target_folder = artm::test::Helpers::getUniqueString();

  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_Format_BagOfWordsUci);
  config.set_target_folder(target_folder);
  config.set_dictionary_file_name("test_parser.dictionary");
  config.set_vocab_file_path("../../../test_data/vocab.parser_test_multiclass.txt");
  config.set_docword_file_path("../../../test_data/docword.parser_test.txt");

  std::shared_ptr< ::artm::DictionaryConfig> dictionary_parsed = ::artm::ParseCollection(config);
  ASSERT_EQ(dictionary_parsed->entry_size(), 3);
  ASSERT_EQ(dictionary_parsed->entry_size(), 3);
  ASSERT_EQ(dictionary_parsed->entry(0).key_token(), "token1");
  ASSERT_EQ(dictionary_parsed->entry(0).class_id(), "class1");
  ASSERT_EQ(dictionary_parsed->entry(1).key_token(), "token2");
  ASSERT_EQ(dictionary_parsed->entry(1).class_id(), "@default_class");
  ASSERT_EQ(dictionary_parsed->entry(2).key_token(), "token3");
  ASSERT_EQ(dictionary_parsed->entry(2).class_id(), "class1");

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
  config.set_dictionary_file_name("test_parser.dictionary");
  config.set_docword_file_path("../../../test_data/vw_data.txt");
  config.set_num_items_per_batch(1);

  std::shared_ptr< ::artm::DictionaryConfig> dictionary_parsed = ::artm::ParseCollection(config);
  ASSERT_EQ(dictionary_parsed->entry_size(), 4);
  EXPECT_EQ(dictionary_parsed->entry(0).key_token(), "alex");
  EXPECT_EQ(dictionary_parsed->entry(0).class_id(), "author");
  EXPECT_EQ(dictionary_parsed->entry(0).token_count(), 3);
  EXPECT_EQ(dictionary_parsed->entry(1).key_token(), "hello");
  EXPECT_EQ(dictionary_parsed->entry(1).class_id(), "@default_class");
  EXPECT_EQ(dictionary_parsed->entry(1).token_count(), 6);
  EXPECT_EQ(dictionary_parsed->entry(2).key_token(), "noname");
  EXPECT_EQ(dictionary_parsed->entry(2).class_id(), "author");
  EXPECT_EQ(dictionary_parsed->entry(2).token_count(), 1);
  EXPECT_EQ(dictionary_parsed->entry(3).key_token(), "world");
  EXPECT_EQ(dictionary_parsed->entry(3).class_id(), "@default_class");
  EXPECT_EQ(dictionary_parsed->entry(3).token_count(), 2);

  try { boost::filesystem::remove_all(target_folder); }
  catch (...) {}
}
