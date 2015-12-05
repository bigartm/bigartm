// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_COLLECTION_PARSER_H_
#define SRC_ARTM_CORE_COLLECTION_PARSER_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "boost/utility.hpp"

#include "artm/messages.pb.h"
#include "artm/core/common.h"

namespace artm {
namespace core {

class CollectionParser : boost::noncopyable {
 public:
  explicit CollectionParser(const ::artm::CollectionParserConfig& config);

  // Parses the collection from disk according to all options,
  // specified in CollectionParserConfig.
  void Parse();

 private:
  struct CollectionParserTokenInfo {
    CollectionParserTokenInfo()
      : keyword(), class_id(DefaultClass), token_weight(), items_count() {}
    explicit CollectionParserTokenInfo(std::string keyword_, ClassId class_id_)
      : keyword(keyword_), class_id(class_id_), token_weight(0.0f), items_count(0) {}
    std::string keyword;
    ClassId class_id;
    float token_weight;
    int items_count;
  };

  typedef std::map<int, CollectionParserTokenInfo> TokenMap;

  class BatchCollector;

  // ParseDocwordBagOfWordsUci is also used to parse MatrixMarket format, because
  // the format of docword file is the same for both.
  void ParseDocwordBagOfWordsUci(TokenMap* token_map);
  void ParseVowpalWabbit();

  TokenMap ParseVocabBagOfWordsUci();
  TokenMap ParseVocabMatrixMarket();

  CollectionParserConfig config_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_COLLECTION_PARSER_H_
