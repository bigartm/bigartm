// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "boost/utility.hpp"

#include "artm/core/common.h"
#include "artm/core/token.h"

namespace artm {
namespace core {

// BatchNameGenerator is a helper class that generates sequential batch names
// (somehting like 000001.batch, 000002.batch, etc.)
class BatchNameGenerator {
 public:
  explicit BatchNameGenerator(int length, bool use_guid_name);
  std::string next_name(const Batch& batch);

 private:
  int length_;
  std::string next_name_;
  bool use_guid_name_;
};

// CollectionParser class is responsible for parsing all text formats, available in BigARTM (UCI Bow and VW parser).
class CollectionParser : boost::noncopyable {
 public:
  explicit CollectionParser(const ::artm::CollectionParserConfig& config);

  // Parses the collection from disk according to all options,
  // specified in CollectionParserConfig.
  CollectionParserInfo Parse();

 private:
  struct CollectionParserTokenInfo {
    CollectionParserTokenInfo()
      : keyword(), class_id(DefaultClass), token_weight(), items_count() { }
    explicit CollectionParserTokenInfo(std::string keyword_, ClassId class_id_)
      : keyword(keyword_), class_id(class_id_), token_weight(0.0f), items_count(0) { }
    std::string keyword;
    ClassId class_id;
    float token_weight;
    int items_count;
  };

  typedef std::map<int, CollectionParserTokenInfo> TokenMap;

  class BatchCollector;

  // ParseDocwordBagOfWordsUci is also used to parse MatrixMarket format, because
  // the format of docword file is the same for both.
  CollectionParserInfo ParseDocwordBagOfWordsUci(TokenMap* token_map);
  CollectionParserInfo ParseVowpalWabbit();

  TokenMap ParseVocabBagOfWordsUci();
  TokenMap ParseVocabMatrixMarket();

  CollectionParserConfig config_;
};

}  // namespace core
}  // namespace artm
