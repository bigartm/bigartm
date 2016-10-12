// Copyright 2015, Additive Regularization of Topic Models.
#include <fstream>

#include "boost/archive/text_woarchive.hpp"
#include "boost/archive/text_wiarchive.hpp"
#include "boost/archive/binary_oarchive.hpp"
#include "boost/archive/binary_iarchive.hpp"

#include "gtest/gtest.h"

#include "artm/core/token.h"
#include "artm/core/dictionary.h"

// http://stackoverflow.com/questions/3015582/direct-boost-serialization-to-char-array
template<class OutputArchive, class OutputStream, class Object>
void write(const Object& obj, std::string tmpfile) {
  OutputStream filestream(tmpfile);
  OutputArchive output_archive(filestream);
  output_archive << obj;
}

template<class InputArchive, class InputStream, class Object>
Object read(std::string tmpfile) {
  Object retval;
  {
    InputStream filestream(tmpfile);
    InputArchive input_archive(filestream);
    input_archive >> retval;
  }

  return retval;
}

template<class OutputArchive, class InputArchive, class OutputStream, class InputStream, class Object>
Object clone(const Object& obj, std::string tmpfile) {
  write<OutputArchive, OutputStream, typename Object>(obj, tmpfile);
  return read<InputArchive, InputStream, typename Object>(tmpfile);
}

template<class Object>
Object text_clone(const Object& obj, std::string tmpfile = "tmpfile.txt") {
  return clone< ::boost::archive::text_woarchive, ::boost::archive::text_wiarchive,
                std::wofstream, std::wifstream, typename Object>(obj, tmpfile);
}

template<class Object>
Object binary_clone(const Object& obj, std::string tmpfile = "tmpfile.bin") {
  return clone< ::boost::archive::binary_oarchive, ::boost::archive::binary_iarchive,
                std::ofstream, std::ifstream, typename Object>(obj, tmpfile);
}

template<class Object>
Object text_read(std::string tmpfile = "tmpfile.txt") {
  return read< ::boost::archive::text_wiarchive, std::wifstream, typename Object>(tmpfile);
}

// artm_tests.exe --gtest_filter=Serializer.Token
TEST(Serializer, Token) {
  ::artm::core::Token token("my class", "my keyword");

  auto token2 = text_clone(token);
  auto token3 = binary_clone(token);

  ASSERT_EQ(token, token2);
  ASSERT_EQ(token, token3);
}

// This test verifies that we can read older versions of the serialization stream
// The file 'token_version0.txt' in test_data must be saved in previous version of BigARTM (at older version).
TEST(Serializer, TokenVersion0) {
  auto token = text_read< ::artm::core::Token>("../../../test_data/serialization/token_version0.txt");
  ASSERT_EQ(token.class_id, "my class");
  ASSERT_EQ(token.keyword, "my keyword");
}

TEST(Serializer, Dictionary) {
  ::artm::DictionaryData dict_tokens, dict_cooc;
  dict_tokens.add_token("token1"); dict_tokens.add_class_id("c1");
  dict_tokens.add_token("token2"); dict_tokens.add_class_id("c2");
  dict_tokens.add_token("token3"); dict_tokens.add_class_id();
  dict_tokens.add_token_value(2.0f); dict_tokens.add_token_value(3.0f); dict_tokens.add_token_value(6.0f);
  dict_cooc.add_cooc_first_index(0);  dict_cooc.add_cooc_second_index(1);   dict_cooc.add_cooc_value(3.4f);
  dict_cooc.add_cooc_first_index(0);  dict_cooc.add_cooc_second_index(2);   dict_cooc.add_cooc_value(1.4f);
  dict_cooc.add_cooc_first_index(2);  dict_cooc.add_cooc_second_index(1);   dict_cooc.add_cooc_value(5.4f);

  ::artm::core::Dictionary dict(dict_tokens);
  dict.Append(dict_cooc);

  auto dict2 = text_clone(dict);
  auto dict3 = binary_clone(dict);

  ASSERT_EQ(dict, dict2);
  ASSERT_EQ(dict, dict3);
}

TEST(Serializer, DictionaryVersion0) {
  auto dict = text_read< ::artm::core::Dictionary>("../../../test_data/serialization/dictionary_version0.txt");
  ASSERT_EQ(dict.size(), 3);
  ASSERT_EQ(*dict.entry(0), ::artm::core::DictionaryEntry(::artm::core::Token("c1", "token1"), 2.0f, 0.0f, 0.0f));
  ASSERT_EQ(*dict.entry(1), ::artm::core::DictionaryEntry(::artm::core::Token("c2", "token2"), 3.0f, 0.0f, 0.0f));
  ASSERT_EQ(*dict.entry(2), ::artm::core::DictionaryEntry(::artm::core::Token("", "token3"), 6.0f, 0.0f, 0.0f));
  ASSERT_EQ(dict.cooc_info(dict.entry(0)->token())->size(), 2);
  ASSERT_EQ(dict.cooc_info(dict.entry(1)->token()), nullptr);
  ASSERT_EQ(dict.cooc_info(dict.entry(2)->token())->size(), 1);
}
