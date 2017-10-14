// Copyright 2017, Additive Regularization of Topic Models.

// Authors: Marina Suvorova (m.dudarenko@gmail.com)
//          Murat Apishev (great-mel@yandex.ru)

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

  // fields of proto messages for all classes
  std::unordered_map<::artm::core::ClassId, float> class_weight_map;
  std::unordered_map<::artm::core::ClassId, double> normalizer_map;
  std::unordered_map<::artm::core::ClassId, double> raw_map;
  std::unordered_map<::artm::core::ClassId, ::google::protobuf::int64> zero_words_map;

  double normalizer = 0.0;
  double raw = 0.0;
  ::google::protobuf::int64 zero_words = 0;

  // choose class_ids policy
  if (config_.class_id_size() == 0) {
    for (int i = 0; (i < args.class_id_size()) && (i < args.class_weight_size()); ++i) {
      class_weight_map.insert(std::make_pair(args.class_id(i), args.class_weight(i)));
      normalizer_map.insert(std::make_pair(args.class_id(i), 0.0));
      raw_map.insert(std::make_pair(args.class_id(i), 0.0));
      zero_words_map.insert(std::make_pair(args.class_id(i), 0));
    }
  } else {
    for (const auto& class_id : config_.class_id()) {
      for (int i = 0; (i < args.class_id_size()) && (i < args.class_weight_size()); ++i) {
        if (class_id == args.class_id(i)) {
          class_weight_map.insert(std::make_pair(args.class_id(i), args.class_weight(i)));
          normalizer_map.insert(std::make_pair(args.class_id(i), 0.0));
          raw_map.insert(std::make_pair(args.class_id(i), 0.0));
          zero_words_map.insert(std::make_pair(args.class_id(i), 0));
          break;
        }
      }
    }
    if (class_weight_map.empty()) {
      LOG(ERROR) << "None of requested classes were presented in model. Score calculation will be skipped";
      return;
    }
  }
  const bool use_class_ids = !class_weight_map.empty();

  // count perplexity normalizer n_d
  for (int token_index = 0; token_index < item.token_weight_size(); ++token_index) {
    if (use_class_ids) {
      ::artm::core::ClassId class_id = token_dict[item.token_id(token_index)].class_id;
      auto class_weight_iter = class_weight_map.find(class_id);
      if (class_weight_iter == class_weight_map.end()) {
        // we should not take tokens without class id weight into consideration
        continue;
      }

      normalizer_map[class_id] += item.token_weight(token_index);
    } else {
      normalizer += item.token_weight(token_index);
    }
  }

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
        LOG(ERROR) << "Perplexity was configured to use UnigramCollectionModel with dictionary " <<
           config_.dictionary_name() << ". This dictionary can't be found.";
        return;
      }
    }
  }

  // count raw values
  std::vector<float> helper_vector(topic_size, 0.0f);
  for (int token_index = 0; token_index < item.token_weight_size(); ++token_index) {
    double sum = 0.0;
    const auto& token = token_dict[item.token_id(token_index)];

    float class_weight = 1.0f;
    if (use_class_ids) {
      auto class_weight_iter = class_weight_map.find(token.class_id);
      if (class_weight_iter == class_weight_map.end()) {
        continue;
      }
      class_weight = class_weight_iter->second;
    }

    float token_weight = class_weight * item.token_weight(token_index);
    if (token_weight == 0.0f) {
      continue;
    }


    int p_wt_token_index = p_wt.token_index(token);
    if (p_wt_token_index != ::artm::core::PhiMatrix::kUndefIndex) {
      p_wt.get(p_wt_token_index, &helper_vector);
      for (int topic_index = 0; topic_index < topic_size; topic_index++) {
        sum += theta[topic_index] * helper_vector[topic_index];
      }
    }
    if (sum == 0.0f) {
      if (use_document_unigram_model) {
        sum = token_weight / (use_class_ids ? normalizer_map[token.class_id] : normalizer);
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
          sum = token_weight / (use_class_ids ? normalizer_map[token.class_id] : normalizer);
        }
      }
      // the presence of class_id in the maps here and below is guaranteed
      ++(use_class_ids ? zero_words_map[token.class_id] : zero_words);
    }
    (use_class_ids ? raw_map[token.class_id] : raw) += token_weight * log(sum);
  }

  // prepare results
  PerplexityScore perplexity_score;
  if (use_class_ids) {
    for (auto iter = normalizer_map.begin(); iter != normalizer_map.end(); ++iter) {
      auto class_id_info = perplexity_score.add_class_id_info();

      class_id_info->set_class_id(iter->first);
      class_id_info->set_normalizer(iter->second);
      class_id_info->set_raw(raw_map[iter->first]);
      class_id_info->set_zero_words(zero_words_map[iter->first]);
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

  bool empty_target = !perplexity_target->class_id_info_size() && !perplexity_target->normalizer();
  bool score_has_class_ids = perplexity_score->class_id_info_size();
  bool target_has_class_ids = empty_target ? score_has_class_ids : perplexity_target->class_id_info_size();
  if (target_has_class_ids != score_has_class_ids) {
    std::stringstream ss;
    ss <<"Inconsistent new content of perplexity score. Old content uses class ids: " << target_has_class_ids;
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(ss.str()));
  }

  double pre_value = 0.0;
  if (target_has_class_ids) {
    for (size_t i = 0; i < perplexity_score->class_id_info_size(); ++i) {
      auto src = perplexity_score->class_id_info(i);

      bool was_added = false;
      for (size_t j = 0; j < perplexity_target->class_id_info_size(); ++j) {
        if (perplexity_score->class_id_info(i).class_id() == perplexity_target->class_id_info(j).class_id()) {
          // update existing class_id info
          auto dst = perplexity_target->mutable_class_id_info(j);
          dst->set_normalizer(dst->normalizer() + src.normalizer());
          dst->set_raw(dst->raw() + src.raw());
          dst->set_zero_words(dst->zero_words() + src.zero_words());

          was_added = true;
          break;
        }
      }

      if (!was_added) {
        // add new class_id info
        auto dst = perplexity_target->add_class_id_info();
        dst->set_class_id(src.class_id());
        dst->set_normalizer(src.normalizer());
        dst->set_raw(src.raw());
        dst->set_zero_words(src.zero_words());
      }
    }

    for (size_t j = 0; j < perplexity_target->class_id_info_size(); ++j) {
      auto score = perplexity_target->class_id_info(j);
      pre_value += score.raw() / score.normalizer();

      VLOG(1) << "class_id=" << score.class_id()
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

    VLOG(1) << "use all class_ids"
            << ", normalizer=" << dst->normalizer()
            << ", raw=" << dst->raw()
            << ", zero_words=" << dst->zero_words();
  }

  perplexity_target->set_value(exp(-pre_value));
}

}  // namespace score
}  // namespace artm
