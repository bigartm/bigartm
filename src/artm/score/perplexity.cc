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
  const Batch& batch,
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
  std::unordered_map<::artm::core::TransactionTypeName, float> transaction_weight_map;
  std::unordered_map<::artm::core::TransactionTypeName, double> normalizer_map;
  std::unordered_map<::artm::core::TransactionTypeName, double> raw_map;
  std::unordered_map<::artm::core::TransactionTypeName, ::google::protobuf::int64> zero_words_map;

  double normalizer = 0.0;
  double raw = 0.0;
  ::google::protobuf::int64 zero_words = 0;

  auto func = [&](const artm::core::TransactionTypeName& name, float value) {  // NOLINT
    transaction_weight_map.emplace(name, value);
    normalizer_map.emplace(name, 0.0);
    raw_map.emplace(name, 0.0);
    zero_words_map.emplace(name, 0);
  };

  if (config_.transaction_typename_size() == 0) {
    for (int i = 0; (i < args.transaction_typename_size()) && (i < args.transaction_weight_size()); ++i) {
      func(args.transaction_typename(i), args.transaction_weight(i));
    }
  } else {
    for (const auto& tt_name : config_.transaction_typename()) {
      for (int i = 0; (i < args.transaction_typename_size()) && (i < args.transaction_weight_size()); ++i) {
        const auto& name = args.transaction_typename(i);
        if (tt_name == name) {
          func(args.transaction_typename(i), args.transaction_weight(i));
          break;
        }
      }
    }
    if (transaction_weight_map.empty()) {
      LOG_FIRST_N(ERROR, 100) << "None of requested transaction typenames are presented in model."
                              << " Score calculation will be skipped";
      return;
    }
  }

  const bool use_tt = !transaction_weight_map.empty();

  std::unordered_map<artm::core::ClassId, float> class_id_to_weight;
  if (config_.class_id_size() == 0) {
    for (int i = 0; (i < args.class_id_size()) && (i < args.class_weight_size()); ++i) {
      class_id_to_weight.emplace(args.class_id(i), args.class_weight(i));
    }
  } else {
    for (const auto& class_id : config_.class_id()) {
      for (int i = 0; (i < args.class_id_size()) && (i < args.class_weight_size()); ++i) {
        if (class_id == args.class_id(i)) {
          class_id_to_weight.emplace(args.class_id(i), args.class_weight(i));
          break;
        }
      }
    }
    if (class_id_to_weight.empty()) {
      LOG_FIRST_N(ERROR, 100) << "None of requested class ids are presented in model."
        << " Score calculation will be skipped";
      return;
    }
  }

  bool use_class_weight = !class_id_to_weight.empty();

  auto t_func = [&](int s_idx, int e_idx) -> float {  // NOLINT
    float transaction_weight = 0.0f;
    for (int idx = s_idx; idx < e_idx; ++idx) {
      const int token_id = item.token_id(idx);
      const float token_weight = item.token_weight(idx);

      float class_weight = 1.0f;
      if (use_class_weight) {
        artm::core::ClassId class_id = batch.class_id(token_id);
        auto iter = class_id_to_weight.find(class_id);
        class_weight = (iter == class_id_to_weight.end()) ? 0.0f : iter->second;
      }

      transaction_weight += (token_weight * class_weight);
    }
    return transaction_weight;
  };

  // count perplexity normalizer n_d
  for (int t_index = 0; t_index < item.transaction_start_index_size() - 1; ++t_index) {
    const int start_index = item.transaction_start_index(t_index);
    const int end_index = item.transaction_start_index(t_index + 1);

    float transaction_weight = t_func(start_index, end_index);
    if (use_tt) {
      float tt_weight = 0.0f;
      const auto& tt_name = batch.transaction_typename(item.transaction_typename_id(t_index));
      auto iter = transaction_weight_map.find(tt_name);
      if (iter != transaction_weight_map.end()) {
        tt_weight = iter->second;
      }
      normalizer_map[tt_name] += tt_weight * transaction_weight;
    } else {
      normalizer += transaction_weight;
    }
  }

  // count raw values
  std::vector<float> helper_vector(topic_size, 0.0f);
  for (int t_index = 0; t_index < item.transaction_start_index_size() - 1; ++t_index) {
    const int start_index = item.transaction_start_index(t_index);
    const int end_index = item.transaction_start_index(t_index + 1);

    double sum = 0.0;
    const auto& tt_name = batch.transaction_typename(item.transaction_typename_id(t_index));
    float transaction_weight = t_func(start_index, end_index);

    float tt_weight = 1.0f;
    if (use_tt) {
      auto iter = transaction_weight_map.find(tt_name);
      if (iter == transaction_weight_map.end()) {
        continue;
      }
      tt_weight = iter->second;
    }

    if (core::isZero(transaction_weight)) {
      continue;
    }

    std::vector<float> phi_values(topic_size, 1.0f);
    for (int token_id = start_index; token_id < end_index; ++token_id) {
      const auto& temp_token = token_dict[item.token_id(token_id)];
      const auto token = artm::core::Token(temp_token.class_id, temp_token.keyword);

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
        sum = transaction_weight / (use_tt ? normalizer_map[tt_name] : normalizer);
      } else {
        sum = 1.0;
        bool failed = true;
        const artm::core::Token* err_token;
        for (int token_id = start_index; token_id < end_index; ++token_id) {
          const auto& temp_token = token_dict[item.token_id(token_id)];
          const auto token = artm::core::Token(temp_token.class_id, temp_token.keyword);

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
          sum = transaction_weight / (use_tt ? normalizer_map[tt_name] : normalizer);
        }
      }
      // the presence of class_id in the maps here and below is guaranteed
      ++(use_tt ? zero_words_map[tt_name] : zero_words);
    }
    (use_tt ? raw_map[tt_name] : raw) += transaction_weight * log(sum);
  }

  // prepare results
  PerplexityScore perplexity_score;
  if (use_tt) {
    for (auto iter = normalizer_map.begin(); iter != normalizer_map.end(); ++iter) {
      auto tt_info = perplexity_score.add_transaction_typename_info();
      tt_info->set_transaction_typename(iter->first);

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

  bool empty_target = !perplexity_target->transaction_typename_info_size() && !perplexity_target->normalizer();
  bool score_has_transactions = perplexity_score->transaction_typename_info_size();
  bool target_has_transactions =
    empty_target ? score_has_transactions : perplexity_target->transaction_typename_info_size();
  if (target_has_transactions != score_has_transactions) {
    std::stringstream ss;
    ss <<"Inconsistent new content of perplexity score. Old content uses transaction types: "
       << target_has_transactions;
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(ss.str()));
  }

  double pre_value = 0.0;
  if (target_has_transactions) {
    for (size_t i = 0; i < perplexity_score->transaction_typename_info_size(); ++i) {
      auto src = perplexity_score->transaction_typename_info(i);

      bool was_added = false;
      for (size_t j = 0; j < perplexity_target->transaction_typename_info_size(); ++j) {
        auto tt1 = perplexity_score->transaction_typename_info(i).transaction_typename();
        auto tt2 = perplexity_target->transaction_typename_info(j).transaction_typename();
        if (tt1 == tt2) {
          // update existing transaction_type info
          auto dst = perplexity_target->mutable_transaction_typename_info(j);
          dst->set_normalizer(dst->normalizer() + src.normalizer());
          dst->set_raw(dst->raw() + src.raw());
          dst->set_zero_words(dst->zero_words() + src.zero_words());

          was_added = true;
          break;
        }
      }

      if (!was_added) {
        // add new transaction_type info
        auto dst = perplexity_target->add_transaction_typename_info();
        dst->set_transaction_typename(src.transaction_typename());
        dst->set_normalizer(src.normalizer());
        dst->set_raw(src.raw());
        dst->set_zero_words(src.zero_words());
      }
    }

    double raw = 0.0;
    double normalizer = 0.0;
    for (size_t j = 0; j < perplexity_target->transaction_typename_info_size(); ++j) {
      auto score = perplexity_target->transaction_typename_info(j);
      raw += score.raw();
      normalizer += score.normalizer();
      VLOG(1) << "transaction_type=" << score.transaction_typename()
              << ", normalizer=" << score.normalizer()
              << ", raw=" << score.raw()
              << ", zero_words=" << score.zero_words();
    }
    pre_value = raw / normalizer;
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
