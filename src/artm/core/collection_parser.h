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
    explicit CollectionParserTokenInfo()
      : keyword(), token_count(), items_count() {}
    explicit CollectionParserTokenInfo(std::string keyword_)
      : keyword(keyword_), token_count(0), items_count(0) {}
    std::string keyword;
    int token_count;
    int items_count;
  };

  typedef std::map<int, CollectionParserTokenInfo> TokenMap;

  class CoocurrenceStatisticsAccumulator {
   public:
    CoocurrenceStatisticsAccumulator(
      const TokenMap& token_info,
      const ::google::protobuf::RepeatedPtrField< ::std::string>& tokens_to_collect);

    void AppendTokenId(int token_id);
    void FlushNewItem();
    void Export(DictionaryConfig* dictionary);

   private:
    const TokenMap& token_info_;
    std::set<std::string> tokens_to_collect_;
    std::map<std::pair<int, int>, int> token_coocurrence_;
    std::vector<int> item_tokens_;
  };

  // ParseDocwordBagOfWordsUci is also used to parse MatrixMarket format, because
  // the format of docword file is the same for both.
  std::shared_ptr<DictionaryConfig> ParseDocwordBagOfWordsUci(TokenMap* token_map);

  TokenMap ParseVocabBagOfWordsUci();
  TokenMap ParseVocabMatrixMarket();

  CollectionParserConfig config_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_COLLECTION_PARSER_H_
