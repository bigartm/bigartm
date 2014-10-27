// Copyright 2014, Additive Regularization of Topic Models.

#include "boost/filesystem.hpp"

#include "gtest/gtest.h"

#include "artm/messages.pb.h"
#include "artm/cpp_interface.h"

// To run this particular test:
// artm_tests.exe --gtest_filter=CollectionParser.*
TEST(CollectionParser, UciBagOfWords) {
  // Clean all .batches files
  if (boost::filesystem::exists("collection_parser_test")) {
    boost::filesystem::recursive_directory_iterator it("collection_parser_test");
    boost::filesystem::recursive_directory_iterator endit;
    while (it != endit) {
      if (boost::filesystem::is_regular_file(*it)) {
        if (it->path().extension() == ".batch" || it->path().extension() == ".dictionary")
          boost::filesystem::remove(*it);
      }

      ++it;
    }
  }

  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_Format_BagOfWordsUci);
  config.set_target_folder("collection_parser_test/");
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
    "collection_parser_test/test_parser.dictionary");
  ASSERT_EQ(dictionary_parsed->entry_size(), dictionary_loaded->entry_size());

  ASSERT_EQ(dictionary_loaded->entry_size(), 3);
  ASSERT_EQ(dictionary_loaded->total_token_count(), 18);
  ASSERT_EQ(dictionary_loaded->total_items_count(), 2);
  ASSERT_EQ(dictionary_loaded->entry(0).key_token(), "token1");
  ASSERT_EQ(dictionary_loaded->entry(0).items_count(), 1);
  ASSERT_EQ(dictionary_loaded->entry(0).token_count(), 5);
  ASSERT_GT(dictionary_loaded->entry(0).value(), 0);
  ASSERT_EQ(dictionary_loaded->entry(1).key_token(), "token2");
  ASSERT_EQ(dictionary_loaded->entry(1).items_count(), 2);
  ASSERT_EQ(dictionary_loaded->entry(1).token_count(), 4);
  ASSERT_EQ(dictionary_loaded->entry(2).key_token(), "token3");
  ASSERT_EQ(dictionary_loaded->entry(2).items_count(), 2);
  ASSERT_EQ(dictionary_loaded->entry(2).token_count(), 9);

  std::shared_ptr< ::artm::DictionaryConfig> cooc_dictionary_loaded = ::artm::LoadDictionary(
    "collection_parser_test/test_parser.cooc.dictionary");
  ASSERT_EQ(cooc_dictionary_loaded->entry_size(), 3);
  ASSERT_EQ(cooc_dictionary_loaded->entry(0).key_token(), "token1~token2");
  ASSERT_EQ(cooc_dictionary_loaded->entry(0).items_count(), 1);
  ASSERT_EQ(cooc_dictionary_loaded->entry(1).key_token(), "token1~token3");
  ASSERT_EQ(cooc_dictionary_loaded->entry(1).items_count(), 1);
  ASSERT_EQ(cooc_dictionary_loaded->entry(2).key_token(), "token2~token3");
  ASSERT_EQ(cooc_dictionary_loaded->entry(2).items_count(), 2);

  boost::filesystem::recursive_directory_iterator it("collection_parser_test");
  boost::filesystem::recursive_directory_iterator endit;
  int batches_count = 0;
  while (it != endit) {
    if (boost::filesystem::is_regular_file(*it) && it->path().extension() == ".batch")
      batches_count++;
    ++it;
  }

  ASSERT_EQ(batches_count, 2);
}

TEST(CollectionParser, ErrorHandling) {
  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_Format_BagOfWordsUci);
  config.set_vocab_file_path("no_such_file.txt");
  config.set_docword_file_path("../../../test_data/docword.parser_test.txt");
  ASSERT_THROW(::artm::ParseCollection(config), artm::DiskReadException);

  config.set_vocab_file_path("../../../test_data/vocab.parser_test.txt");
  config.set_docword_file_path("no_such_file");
  ASSERT_THROW(::artm::ParseCollection(config), artm::DiskReadException);

  ASSERT_THROW(::artm::LoadDictionary("no_such_file"), artm::DiskReadException);
}

TEST(CollectionParser, MatrixMarket) {
  // Clean all .batches files
  if (boost::filesystem::exists("collection_parser_test")) {
    boost::filesystem::recursive_directory_iterator it("collection_parser_test");
    boost::filesystem::recursive_directory_iterator endit;
    while (it != endit) {
      if (boost::filesystem::is_regular_file(*it)) {
        if (it->path().extension() == ".batch" || it->path().extension() == ".dictionary")
          boost::filesystem::remove(*it);
      }

      ++it;
    }
  }

  ::artm::CollectionParserConfig config;
  config.set_format(::artm::CollectionParserConfig_Format_MatrixMarket);
  config.set_target_folder("collection_parser_test/");
  config.set_num_items_per_batch(10000);
  config.set_vocab_file_path("../../../test_data/deerwestere.txt");
  config.set_docword_file_path("../../../test_data/deerwestere.mm");
  config.set_dictionary_file_name("test_parser.dictionary");

  std::shared_ptr< ::artm::DictionaryConfig> dictionary_parsed = ::artm::ParseCollection(config);
  ASSERT_EQ(dictionary_parsed->entry_size(), 12);
}
