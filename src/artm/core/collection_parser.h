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
  // Returns a dictionary that lists all unique tokens occured in the collection.
  // Each DictionaryEntry from dictionary will contain key_token
  // and some additional statistics like the number of term occurrences in the collection.
  std::shared_ptr<DictionaryConfig> Parse();

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

  class CoocurrenceStatisticsAccumulator {
   public:
    CoocurrenceStatisticsAccumulator(
      const TokenMap& token_info,
      const ::google::protobuf::RepeatedPtrField< ::std::string>& tokens_to_collect,
      const ::google::protobuf::RepeatedPtrField< ::std::string>& class_ids_to_collect);

    void AppendTokenId(int token_id);
    void FlushNewItem();
    void Export(std::shared_ptr<DictionaryConfig> dictionary);

   private:
    const TokenMap& token_info_;
    std::set<Token> tokens_to_collect_;
    std::map<std::pair<int, int>, int> token_coocurrence_;
    std::vector<int> item_tokens_;
  };
  class BatchCollector;

  // ParseDocwordBagOfWordsUci is also used to parse MatrixMarket format, because
  // the format of docword file is the same for both.
  std::shared_ptr<DictionaryConfig> ParseDocwordBagOfWordsUci(TokenMap* token_map);
  std::shared_ptr<DictionaryConfig> ParseVowpalWabbit();

  std::shared_ptr<DictionaryConfig> ParseCooccurrenceData(TokenMap* token_map);

  TokenMap ParseVocabBagOfWordsUci();
  TokenMap ParseVocabMatrixMarket();

  CollectionParserConfig config_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_COLLECTION_PARSER_H_
