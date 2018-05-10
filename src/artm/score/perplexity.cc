// Copyright 2017, Additive Regularization of Topic Models.

// Authors: Marina Suvorova (m.dudarenko@gmail.com)
//          Murat Apishev (great-mel@yandex.ru)

#include <cmath>
#include <map>
#include <algorithm>
#include <sstream>

#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/score/perplexity.h"

namespace artm {
namespace score {

Perplexity::Perplexity(const ScoreConfig& config) : ScoreCalculatorInterface(config) {
  config_ = ParseConfig<PerplexityScoreConfig>();
  std::stringstream ss;
  ss << ": model_type=" << config_.model_type();
  if (config_.has_dictionary_name()) {
    ss << ", dictionary_name=" << config_.dictionary_name();
  }
  LOG(INFO) << "Perplexity score calculator created" << ss.str();
}

void Perplexity::AppendScore(
    const Item& item,
    const std::vector<artm::core::Token>& token_dict,
    const artm::core::PhiMatrix& p_wt,
    const artm::ProcessBatchesArgs& args,
    const std::vector<float>& theta,
    Score* score) {
  const int topic_size = p_wt.topic_size();

  // check dictionary existence for replacing zero pwt sums
  std::shared_ptr<core::Dictionary> dictionary_ptr = nullptr;
  if (config_.has_dictionary_name()) {
    dictionary_ptr = dictionary(config_.dictionary_name());
  }

  bool use_document_unigram_model = true;
  if (config_.has_model_type()) {
    if (config_.model_type() == PerplexityScoreConfig_Type_UnigramCollectionModel) {
      if (dictionary_ptr) {
        use_document_unigram_model = false;
      } else {
        LOG_FIRST_N(ERROR, 100) << "Perplexity was configured to use UnigramCollectionModel with dictionary "
                                << config_.dictionary_name() << ". This dictionary can't be found.";
        return;
      }
    }
  }

  // fields of proto messages for all classes
  std::unordered_map<::artm::core::TransactionType, float, artm::core::TransactionHasher> transaction_weight_map;
  std::unordered_map<::artm::core::TransactionType, double, artm::core::TransactionHasher> normalizer_map;
  std::unordered_map<::artm::core::TransactionType, double, artm::core::TransactionHasher> raw_map;
  std::unordered_map<::artm::core::TransactionType,
    ::google::protobuf::int64, artm::core::TransactionHasher> zero_words_map;

  double normalizer = 0.0;
  double raw = 0.0;
  ::google::protobuf::int64 zero_words = 0;

  if (config_.transaction_type_size() == 0) {
    for (int i = 0; (i < args.transaction_type_size()) && (i < args.transaction_weight_size()); ++i) {
      const auto& tmp = ::artm::core::TransactionType(args.transaction_type(i));
      transaction_weight_map.insert(std::make_pair(tmp, args.transaction_weight(i)));
      normalizer_map.insert(std::make_pair(tmp, 0.0));
      raw_map.insert(std::make_pair(tmp, 0.0));
      zero_words_map.insert(std::make_pair(tmp, 0));
    }
  } else {
    for (const auto& tt : config_.transaction_type()) {
      for (int i = 0; (i < args.transaction_type_size()) && (i < args.transaction_weight_size()); ++i) {
        const auto& tmp = ::artm::core::TransactionType(args.transaction_type(i));
        if (::artm::core::TransactionType(tt) == tmp) {
          transaction_weight_map.insert(std::make_pair(tmp, args.transaction_weight(i)));
          normalizer_map.insert(std::make_pair(tmp, 0.0));
          raw_map.insert(std::make_pair(tmp, 0.0));
          zero_words_map.insert(std::make_pair(tmp, 0));
          break;
        }
      }
    }
    if (transaction_weight_map.empty()) {
      LOG_FIRST_N(ERROR, 100) << "None of requested transaction types were presented in model."
                              << " Score calculation will be skipped";
      return;
    }
  }

  const bool use_tt = !transaction_weight_map.empty();

  // count perplexity normalizer n_d
  for (int token_index = 0; token_index < item.transaction_start_index_size(); ++token_index) {
    if (use_tt) {
      const int start_index = item.transaction_start_index(token_index);
      const int end_index = (token_index + 1) < item.transaction_start_index_size() ?
                            item.transaction_start_index(token_index + 1) :
                            item.transaction_token_id_size();
      std::string str;
      for (int token_id = start_index; token_id < end_index; ++token_id) {
        const auto& tmp = token_dict[item.transaction_token_id(token_id)].class_id;
        str += (token_id == start_index) ? tmp : artm::core::TransactionSeparator + tmp;
      }

      artm::core::TransactionType tt(str);
      auto iter = transaction_weight_map.find(tt);
      if (iter == transaction_weight_map.end()) {
        // we should not take tokens without transaction type weight into consideration
        continue;
      }
      float tt_weight = iter->second;
      normalizer_map[tt] += tt_weight * item.token_weight(token_index);
    } else {
      normalizer += item.token_weight(token_index);
    }
  }

  // count raw values
  std::vector<float> helper_vector(topic_size, 0.0f);
  for (int token_index = 0; token_index < item.transaction_start_index_size(); ++token_index) {
    double sum = 0.0;
    const int start_index = item.transaction_start_index(token_index);
    const int end_index = (token_index + 1) < item.transaction_start_index_size() ?
                          item.transaction_start_index(token_index + 1) :
                          item.transaction_token_id_size();
    std::string str;
    for (int token_id = start_index; token_id < end_index; ++token_id) {
      auto& tmp = token_dict[item.transaction_token_id(token_id)].class_id;
      str += (token_id == start_index) ? tmp : artm::core::TransactionSeparator + tmp;
    }
    artm::core::TransactionType tt(str);

    float tt_weight = 1.0f;
    if (use_tt) {
      auto iter = transaction_weight_map.find(tt);
      if (iter == transaction_weight_map.end()) {
        continue;
      }
      tt_weight = iter->second;
    }

    float token_weight = tt_weight * item.token_weight(token_index);
    if (core::isZero(token_weight)) {
      continue;
    }

    std::vector<float> phi_values(topic_size, 1.0f);
    for (int token_id = start_index; token_id < end_index; ++token_id) {
      const auto& temp_token = token_dict[item.transaction_token_id(token_id)];
      const auto token = artm::core::Token(temp_token.class_id, temp_token.keyword, tt);

      int p_wt_token_index = p_wt.token_index(token);
      if (p_wt_token_index == ::artm::core::PhiMatrix::kUndefIndex) {
        // ignore tokens that doe not belong to the model
        continue;
      }

      p_wt.get(p_wt_token_index, &helper_vector);
      for (int topic_index = 0; topic_index < topic_size; topic_index++) {
        phi_values[topic_index] *= helper_vector[topic_index];
      }
    }

    for (int topic_index = 0; topic_index < topic_size; topic_index++) {
      sum += theta[topic_index] * phi_values[topic_index];
    }

    if (core::isZero(sum)) {
      if (use_document_unigram_model) {
        sum = token_weight / (use_tt ? normalizer_map[tt] : normalizer);
      } else {
        sum = 1.0;
        bool failed = true;
        const artm::core::Token* err_token;
        for (int token_id = start_index; token_id < end_index; ++token_id) {
          const auto& temp_token = token_dict[item.transaction_token_id(token_id)];
          const auto token = artm::core::Token(temp_token.class_id, temp_token.keyword, tt);

          auto entry_ptr = dictionary_ptr->entry(token);
          if (entry_ptr != nullptr && entry_ptr->token_value()) {
            sum *= entry_ptr->token_value();
          } else {
            err_token = &token;
            break;
          }
          if (token_id == end_index - 1) {
            failed = false;
          }
        }

        if (failed) {
          LOG_FIRST_N(WARNING, 100)
            << "Error in perplexity dictionary for token " << err_token->keyword << ", class " << err_token->class_id
            << " (and potentially for other tokens)"
            << ". Verify that the token exists in the dictionary and it's value > 0. "
            << "Document unigram model will be used for this token "
            << "(and for all other tokens under the same conditions).";
          sum = token_weight / (use_tt ? normalizer_map[tt] : normalizer);
        }
      }
      // the presence of class_id in the maps here and below is guaranteed
      ++(use_tt ? zero_words_map[tt] : zero_words);
    }
    (use_tt ? raw_map[tt] : raw) += token_weight * log(sum);
  }

  // prepare results
  PerplexityScore perplexity_score;
  if (use_tt) {
    for (auto iter = normalizer_map.begin(); iter != normalizer_map.end(); ++iter) {
      auto tt_info = perplexity_score.add_transaction_type_info();
      tt_info->set_transaction_type(iter->first.AsString());

      tt_info->set_normalizer(iter->second);
      tt_info->set_raw(raw_map[iter->first]);
      tt_info->set_zero_words(zero_words_map[iter->first]);
    }
  } else {
    perplexity_score.set_normalizer(normalizer);
    perplexity_score.set_raw(raw);
    perplexity_score.set_zero_words(zero_words);
  }

  AppendScore(perplexity_score, score);
}

std::shared_ptr<Score> Perplexity::CreateScore() {
  VLOG(1) << "Perplexity::CreateScore()";
  return std::make_shared<PerplexityScore>();
}

void Perplexity::AppendScore(const Score& score, Score* target) {
  std::string error_message = "Unable downcast Score to PerplexityScore";
  const PerplexityScore* perplexity_score = dynamic_cast<const PerplexityScore*>(&score);
  if (perplexity_score == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(error_message));
  }

  PerplexityScore* perplexity_target = dynamic_cast<PerplexityScore*>(target);
  if (perplexity_target == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(error_message));
  }

  bool empty_target = !perplexity_target->transaction_type_info_size() && !perplexity_target->normalizer();
  bool score_has_transactions = perplexity_score->transaction_type_info_size();
  bool target_has_transactions =
    empty_target ? score_has_transactions : perplexity_target->transaction_type_info_size();
  if (target_has_transactions != score_has_transactions) {
    std::stringstream ss;
    ss <<"Inconsistent new content of perplexity score. Old content uses transaction types: "
       << target_has_transactions;
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(ss.str()));
  }

  double pre_value = 0.0;
  if (target_has_transactions) {
    for (size_t i = 0; i < perplexity_score->transaction_type_info_size(); ++i) {
      auto src = perplexity_score->transaction_type_info(i);

      bool was_added = false;
      for (size_t j = 0; j < perplexity_target->transaction_type_info_size(); ++j) {
        auto tt1 = ::artm::core::TransactionType(
          perplexity_score->transaction_type_info(i).transaction_type());
        auto tt2 = ::artm::core::TransactionType(
          perplexity_target->transaction_type_info(j).transaction_type());
        if (tt1 == tt2) {
          // update existing transaction_type info
          auto dst = perplexity_target->mutable_transaction_type_info(j);
          dst->set_normalizer(dst->normalizer() + src.normalizer());
          dst->set_raw(dst->raw() + src.raw());
          dst->set_zero_words(dst->zero_words() + src.zero_words());

          was_added = true;
          break;
        }
      }

      if (!was_added) {
        // add new transaction_type info
        auto dst = perplexity_target->add_transaction_type_info();
        dst->set_transaction_type(src.transaction_type());
        dst->set_normalizer(src.normalizer());
        dst->set_raw(src.raw());
        dst->set_zero_words(src.zero_words());
      }
    }

    for (size_t j = 0; j < perplexity_target->transaction_type_info_size(); ++j) {
      auto score = perplexity_target->transaction_type_info(j);
      pre_value += score.raw() / score.normalizer();
      VLOG(1) << "transaction_type=" << score.transaction_type()
              << ", normalizer=" << score.normalizer()
              << ", raw=" << score.raw()
              << ", zero_words=" << score.zero_words();
    }
  } else {
    auto src = perplexity_score;
    auto dst = perplexity_target;

    dst->set_normalizer(dst->normalizer() + src->normalizer());
    dst->set_raw(dst->raw() + src->raw());
    dst->set_zero_words(dst->zero_words() + src->zero_words());

    pre_value = dst->raw() / dst->normalizer();

    VLOG(1) << "use all transaction_types"
            << ", normalizer=" << dst->normalizer()
            << ", raw=" << dst->raw()
            << ", zero_words=" << dst->zero_words();
  }

  perplexity_target->set_value(exp(-pre_value));
}

}  // namespace score
}  // namespace artm
