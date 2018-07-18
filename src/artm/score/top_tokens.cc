// Copyright 2017, Additive Regularization of Topic Models.

// Authors: Marina Suvorova (m.dudarenko@gmail.com)
//          Murat Apishev (great-mel@yandex.ru)

#include <algorithm>
#include <utility>

#include "artm/core/dictionary.h"
#include "artm/core/exceptions.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/score/top_tokens.h"

namespace artm {
namespace score {

std::shared_ptr<Score> TopTokens::CalculateScore(const artm::core::PhiMatrix& p_wt) {
  const int topic_size = p_wt.topic_size();
  const int token_size = p_wt.token_size();

  std::shared_ptr<core::Dictionary> dictionary_ptr = nullptr;
  if (config_.has_cooccurrence_dictionary_name()) {
    dictionary_ptr = dictionary(config_.cooccurrence_dictionary_name());
  }
  bool count_coherence = dictionary_ptr != nullptr;

  std::vector<int> topic_ids;
  google::protobuf::RepeatedPtrField<std::string> topic_name = p_wt.topic_name();
  if (config_.topic_name_size() == 0) {
    for (int i = 0; i < topic_size; ++i) {
      topic_ids.push_back(i);
    }
  } else {
    for (int i = 0; i < config_.topic_name_size(); ++i) {
      int index = ::artm::core::repeated_field_index_of(topic_name, config_.topic_name(i));
      if (index == -1) {
        BOOST_THROW_EXCEPTION(::artm::core::InvalidOperation(
          "Topic with name '" + config_.topic_name(i) + "' not found in the model"));
      }
      topic_ids.push_back(index);
    }
  }

  ::artm::core::ClassId class_id = ::artm::core::DefaultClass;
  if (config_.has_class_id()) {
    class_id = config_.class_id();
  }

  if (count_coherence) {
    LOG(ERROR) << "Coherence computation in TopTokens score does not support transactions!";
    return false;
  }

  std::vector<artm::core::Token> tokens;
  for (int token_index = 0; token_index < token_size; token_index++) {
    auto token = p_wt.token(token_index);
    if (token.class_id == class_id) {
      tokens.push_back(token);
    }
  }

  TopTokensScore* top_tokens_score = new TopTokensScore();
  std::shared_ptr<Score> retval(top_tokens_score);
  int num_entries = 0;

  float average_coherence = 0.0f;
  auto coherence = top_tokens_score->mutable_coherence();
  for (unsigned i = 0; i < topic_ids.size(); ++i) {
    std::vector<std::pair<float, int>> p_wt_local;
    p_wt_local.reserve(tokens.size());

    for (int token_index = 0; token_index < token_size; token_index++) {
      const auto& token = p_wt.token(token_index);
      if (token.class_id != class_id) {
        continue;
      }

      float weight = p_wt.get(token_index, topic_ids[i]);
      p_wt_local.push_back(std::pair<float, int>(weight, p_wt_local.size()));
    }

    std::sort(p_wt_local.begin(), p_wt_local.end());

    int first_index = p_wt_local.size() - 1;
    int last_index = (p_wt_local.size() - config_.num_tokens());
    if (last_index < 0) {
      last_index = 0;
    }

    std::vector<core::Token> tokens_for_coherence;
    for (int token_index = first_index; token_index >= last_index; token_index--) {
      ::artm::core::Token token = tokens[p_wt_local[token_index].second];
      float weight = p_wt_local[token_index].first;
      if (weight < config_.eps()) {
        continue;
      }

      top_tokens_score->add_token(token.keyword);
      top_tokens_score->add_weight(weight);
      top_tokens_score->add_topic_index(topic_ids[i]);
      top_tokens_score->add_topic_name(topic_name.Get(topic_ids[i]));
      ++num_entries;

      if (count_coherence && weight > 0.0f) {
        tokens_for_coherence.push_back(token);
      }
    }

    if (count_coherence) {
      float topic_coherence = dictionary_ptr->CountTopicCoherence(tokens_for_coherence);
      average_coherence += topic_coherence;
      coherence->Add(topic_coherence);
    }
  }

  top_tokens_score->set_average_coherence(
    average_coherence > 0.0f ? average_coherence / topic_ids.size() : average_coherence);

  top_tokens_score->set_num_entries(num_entries);
  return retval;
}

}  // namespace score
}  // namespace artm
