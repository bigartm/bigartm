// Copyright 2014, Additive Regularization of Topic Models.

// Author: Marina Suvorova (m.dudarenko@gmail.com)

#include <cmath>
#include <map>
#include <algorithm>
#include <sstream>

#include "artm/core/exceptions.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/score/perplexity.h"

namespace artm {
namespace score {

Perplexity::Perplexity(const ScoreConfig& config) : ScoreCalculatorInterface(config) {
  config_ = ParseConfig<PerplexityScoreConfig>();
  std::stringstream ss;
  ss << ": model_type=" << config_.model_type();
  if (config_.has_dictionary_name())
    ss << ", dictionary_name=" << config_.dictionary_name();
  LOG(INFO) << "Perplexity score calculator created" << ss.str();
}

void Perplexity::AppendScore(
    const Item& item,
    const std::vector<artm::core::Token>& token_dict,
    const artm::core::PhiMatrix& p_wt,
    const artm::ProcessBatchesArgs& args,
    const std::vector<float>& theta,
    Score* score) {
  int topic_size = p_wt.topic_size();

  // the following code counts perplexity
  bool use_classes_from_model = false;
  if (config_.class_id_size() == 0) use_classes_from_model = true;

  std::map< ::artm::core::ClassId, float> class_weights;
  if (use_classes_from_model) {
    for (int i = 0; (i < args.class_id_size()) && (i < args.class_weight_size()); ++i)
      class_weights.insert(std::make_pair(args.class_id(i), args.class_weight(i)));
  } else {
    for (auto& class_id : config_.class_id()) {
      for (int i = 0; (i < args.class_id_size()) && (i < args.class_weight_size()); ++i)
        if (class_id == args.class_id(i)) {
          class_weights.insert(std::make_pair(args.class_id(i), args.class_weight(i)));
          break;
        }
    }
  }
  bool use_class_id = !class_weights.empty();

  float n_d = 0;
  for (int token_index = 0; token_index < item.token_weight_size(); ++token_index) {
    float class_weight = 1.0f;
    if (use_class_id) {
      ::artm::core::ClassId class_id = token_dict[item.token_id(token_index)].class_id;
      auto iter = class_weights.find(class_id);
      if (iter == class_weights.end())
        continue;
      class_weight = iter->second;
    }

    n_d += class_weight * item.token_weight(token_index);
  }

  ::google::protobuf::int64 zero_words = 0;
  double normalizer = 0;
  double raw = 0;

  std::shared_ptr<core::Dictionary> dictionary_ptr = nullptr;
  if (config_.has_dictionary_name())
    dictionary_ptr = dictionary(config_.dictionary_name());
  bool has_dictionary = dictionary_ptr != nullptr;

  bool use_document_unigram_model = true;
  if (config_.has_model_type()) {
    if (config_.model_type() == PerplexityScoreConfig_Type_UnigramCollectionModel) {
      if (has_dictionary) {
        use_document_unigram_model = false;
      } else {
        LOG(ERROR) << "Perplexity was configured to use UnigramCollectionModel with dictionary " <<
           config_.dictionary_name() << ". This dictionary can't be found.";
        return;
      }
    }
  }

  for (int token_index = 0; token_index < item.token_weight_size(); ++token_index) {
    double sum = 0.0;
    const artm::core::Token& token = token_dict[item.token_id(token_index)];

    float class_weight = 1.0f;
    if (use_class_id) {
      auto iter = class_weights.find(token.class_id);
      if (iter == class_weights.end())
        continue;
      class_weight = iter->second;
    }

    float token_weight = class_weight * item.token_weight(token_index);
    if (token_weight == 0.0f) continue;

    int p_wt_token_index = p_wt.token_index(token);
    if (p_wt_token_index != ::artm::core::PhiMatrix::kUndefIndex) {
      for (int topic_index = 0; topic_index < topic_size; topic_index++) {
        sum += theta[topic_index] * p_wt.get(p_wt_token_index, topic_index);
      }
    }
    if (sum == 0.0) {
      if (use_document_unigram_model) {
        sum = token_weight / n_d;
      } else {
        auto entry_ptr = dictionary_ptr->entry(token);
        bool failed = true;
        if (entry_ptr != nullptr && entry_ptr->token_value()) {
          sum = entry_ptr->token_value();
          failed = false;
        }
        if (failed) {
          LOG_FIRST_N(WARNING, 1)
                    << "Error in perplexity dictionary for token " << token.keyword << ", class " << token.class_id
                    << " (and potentially for other tokens)"
                    << ". Verify that the token exists in the dictionary and it's value > 0. "
                    << "Document unigram model will be used for this token "
                    << "(and for all other tokens under the same conditions).";
          sum = token_weight / n_d;
        }
      }
      zero_words++;
    }

    normalizer += token_weight;
    raw        += token_weight * log(sum);
  }

  // prepare results
  PerplexityScore perplexity_score;
  perplexity_score.set_normalizer(normalizer);
  perplexity_score.set_raw(raw);
  perplexity_score.set_zero_words(zero_words);
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

  perplexity_target->set_normalizer(perplexity_target->normalizer() +
                                    perplexity_score->normalizer());
  perplexity_target->set_raw(perplexity_target->raw() +
                             perplexity_score->raw());
  perplexity_target->set_zero_words(perplexity_target->zero_words() +
                                    perplexity_score->zero_words());
  perplexity_target->set_value(exp(- perplexity_target->raw() / perplexity_target->normalizer()));

  VLOG(1) << "normalizer=" << perplexity_target->normalizer()
          << ", raw=" << perplexity_target->raw()
          << ", zero_words=" << perplexity_target->zero_words();
}

}  // namespace score
}  // namespace artm
