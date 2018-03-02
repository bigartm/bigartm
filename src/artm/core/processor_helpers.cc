// Copyright 2018, Additive Regularization of Topic Models.

#include "artm/core/processor_helpers.h"

namespace artm {
namespace core {

void ProcessorHelpers::CreateThetaCacheEntry(ThetaMatrix* new_cache_entry_ptr,
                                             LocalThetaMatrix<float>* theta_matrix,
                                             const Batch& batch,
                                             const PhiMatrix& p_wt,
                                             const ProcessBatchesArgs& args) {
  if (new_cache_entry_ptr == nullptr) {
    return;
  }

  const int topic_size = p_wt.topic_size();
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    const Item& item = batch.item(item_index);
    new_cache_entry_ptr->add_item_id(item.id());
    new_cache_entry_ptr->add_item_title(item.has_title() ? item.title() : std::string());
    new_cache_entry_ptr->add_item_weights();
  }

  if (!args.has_predict_transaction_type() && !args.has_predict_class_id()) {
    for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
        new_cache_entry_ptr->mutable_item_weights(item_index)->add_value((*theta_matrix)(topic_index, item_index));
      }
    }
  } else {
    const bool predict_tt = args.has_predict_transaction_type();
    const bool predict_class_id = args.has_predict_class_id();
    new_cache_entry_ptr->clear_topic_name();
    for (int token_index = 0; token_index < p_wt.token_size(); token_index++) {
      const Token& token = p_wt.token(token_index);
      if ((predict_class_id && token.class_id != args.predict_class_id() ||
          (predict_tt && token.transaction_type != TransactionType(args.predict_transaction_type())))) {
        continue;
      }

      new_cache_entry_ptr->add_topic_name(token.keyword);
      for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
        float weight = 0.0;
        for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
          weight += (*theta_matrix)(topic_index, item_index) * p_wt.get(token_index, topic_index);
        }
        new_cache_entry_ptr->mutable_item_weights(item_index)->add_value(weight);
      }
    }
  }
}

void ProcessorHelpers::CreatePtdwCacheEntry(ThetaMatrix* new_cache_entry_ptr,
                                            LocalPhiMatrix<float>* ptdw_matrix,
                                            const Batch& batch,
                                            int item_index,
                                            int topic_size) {
  if (new_cache_entry_ptr == nullptr) {
    return;
  }

  const Item& item = batch.item(item_index);
  for (int token_index = 0; token_index < ptdw_matrix->num_tokens(); ++token_index) {
    new_cache_entry_ptr->add_item_id(item.id());
    new_cache_entry_ptr->add_item_title(item.has_title() ? item.title() : std::string());
    auto non_zero_topic_values = new_cache_entry_ptr->add_item_weights();
    auto non_zero_topic_indices = new_cache_entry_ptr->add_topic_indices();

    for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
      float value = ptdw_matrix->operator()(token_index, topic_index);
      if (std::fabs(value) > kProcessorEps) {
        // store not-null values p(t|d,w) for given d and w
        non_zero_topic_values->add_value(value);
        // store indices of these not-null values
        non_zero_topic_indices->add_value(topic_index);
      }
    }
  }
}

std::shared_ptr<LocalThetaMatrix<float>> ProcessorHelpers::InitializeTheta(int topic_size,
                                                                           const Batch& batch,
                                                                           const ProcessBatchesArgs& args,
                                                                           const ThetaMatrix* cache) {
  auto Theta = std::make_shared<LocalThetaMatrix<float>>(topic_size, batch.item_size());

  Theta->InitializeZeros();

  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    int index_of_item = -1;
    if ((cache != nullptr) && args.reuse_theta()) {
      index_of_item = repeated_field_index_of(cache->item_title(),
        batch.item(item_index).title());
    }

    if ((index_of_item != -1) && args.reuse_theta()) {
      const FloatArray& old_thetas = cache->item_weights(index_of_item);
      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
        (*Theta)(topic_index, item_index) = old_thetas.value(topic_index);
      }
    } else {
      if (args.use_random_theta()) {
        size_t seed = 0;
        boost::hash_combine(seed, std::hash<std::string>()(batch.id()));
        boost::hash_combine(seed, std::hash<int>()(item_index));
        std::vector<float> theta_values = Helpers::GenerateRandomVector(topic_size, seed);
        for (int iTopic = 0; iTopic < topic_size; ++iTopic) {
          (*Theta)(iTopic, item_index) = theta_values[iTopic];
        }
      } else {
        const float default_theta = 1.0f / topic_size;
        for (int iTopic = 0; iTopic < topic_size; ++iTopic) {
          (*Theta)(iTopic, item_index) = default_theta;
        }
      }
    }
  }

  return Theta;
}

std::shared_ptr<LocalPhiMatrix<float>>
ProcessorHelpers::InitializePhi(const Batch& batch, const ::artm::core::PhiMatrix& p_wt) {
  bool phi_is_empty = true;
  int topic_size = p_wt.topic_size();
  auto phi_matrix = std::make_shared<LocalPhiMatrix<float>>(batch.token_size(), topic_size);
  phi_matrix->InitializeZeros();
  for (int token_index = 0; token_index < batch.token_size(); ++token_index) {
    Token token = Token(batch.class_id(token_index), batch.token(token_index));

    int p_wt_token_index = p_wt.token_index(token);
    if (p_wt_token_index != ::artm::core::PhiMatrix::kUndefIndex) {
      phi_is_empty = false;
      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
        float value = p_wt.get(p_wt_token_index, topic_index);
        if (value < kProcessorEps) {
          // Reset small values to 0.0 to avoid performance hit.
          // http://en.wikipedia.org/wiki/Denormal_number#Performance_issues
          // http://stackoverflow.com/questions/13964606/inconsistent-multiplication-performance-with-floats
          value = 0.0f;
        }
        (*phi_matrix)(token_index, topic_index) = value;
      }
    }
  }

  if (phi_is_empty) {
    return nullptr;
  }

  return phi_matrix;
}

void ProcessorHelpers::CreateRegularizerAgents(const Batch& batch,
                                               const ProcessBatchesArgs& args,
                                               Instance* instance,
                                               RegularizeThetaAgentCollection* theta_agents,
                                               RegularizePtdwAgentCollection* ptdw_agents) {
  for (int reg_index = 0; reg_index < args.regularizer_name_size(); ++reg_index) {
    auto& reg_name = args.regularizer_name(reg_index);
    float tau = args.regularizer_tau(reg_index);
    auto regularizer = instance->regularizers()->get(reg_name);
    if (regularizer == nullptr) {
      LOG(ERROR) << "Theta Regularizer with name <" << reg_name << "> does not exist.";
      continue;
    }

    if (theta_agents != nullptr) {
      theta_agents->AddAgent(regularizer->CreateRegularizeThetaAgent(batch, args, tau));
    }

    if (ptdw_agents != nullptr) {
      ptdw_agents->AddAgent(regularizer->CreateRegularizePtdwAgent(batch, args, tau));
    }
  }

  if (theta_agents != nullptr) {
    theta_agents->AddAgent(std::make_shared<NormalizeThetaAgent>());
  }
}

std::shared_ptr<CsrMatrix<float>> ProcessorHelpers::InitializeSparseNdw(const Batch& batch,
                                                                        const ProcessBatchesArgs& args) {
  std::vector<float> n_dw_val;
  std::vector<int> n_dw_row_ptr;
  std::vector<int> n_dw_col_ind;

  bool use_weights = false;
  std::unordered_map<TransactionType, float, TransactionHasher> tt_to_weight;
  if (args.transaction_type_size() != 0) {
    use_weights = true;
    for (int i = 0; i < args.transaction_type_size(); ++i) {
      tt_to_weight.insert(std::make_pair(TransactionType(args.transaction_type(i)),
                                         args.transaction_weight(i)));
    }
  }

  // For sparse case
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
    const Item& item = batch.item(item_index);

    for (int token_index = 0; token_index < item.transaction_start_index_size(); ++token_index) {
      const int start_index = item.transaction_start_index(token_index);
      const int end_index = (token_index + 1) < item.transaction_start_index_size() ?
        item.transaction_start_index(token_index + 1) :
        item.transaction_token_id_size();

      for (int idx = start_index; idx < end_index; ++idx) {
        const int token_id = item.transaction_token_id(idx);
        float tt_weight = 1.0f;
        if (use_weights) {
          ClassId class_id = batch.class_id(token_id);
          auto iter = tt_to_weight.find(TransactionType(class_id));
          tt_weight = (iter == tt_to_weight.end()) ? 0.0f : iter->second;
        }
        const float token_weight = item.token_weight(token_index);
        n_dw_val.push_back(tt_weight * token_weight);
        n_dw_col_ind.push_back(token_id);
      }
    }
  }
  n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
  return std::make_shared<CsrMatrix<float>>(batch.token_size(), &n_dw_val, &n_dw_row_ptr, &n_dw_col_ind);
}

std::shared_ptr<Score> ProcessorHelpers::CalcScores(ScoreCalculatorInterface* score_calc,
                                                    const Batch& batch,
                                                    const PhiMatrix& p_wt,
                                                    const ProcessBatchesArgs& args,
                                                    const LocalThetaMatrix<float>& theta_matrix) {
  if (!score_calc->is_cumulative()) {
    return nullptr;
  }

  std::vector<Token> batch_token_dict;
  for (int token_index = 0; token_index < batch.token_size(); ++token_index) {
    batch_token_dict.push_back(Token(batch.class_id(token_index), batch.token(token_index)));
  }

  std::shared_ptr<Score> score = score_calc->CreateScore();
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    const Item& item = batch.item(item_index);

    std::vector<float> theta_vec;
    assert(theta_matrix.num_topics() == p_wt.topic_size());
    for (int topic_index = 0; topic_index < theta_matrix.num_topics(); ++topic_index) {
      theta_vec.push_back(theta_matrix(topic_index, item_index));
    }

    score_calc->AppendScore(item, batch_token_dict, p_wt, args, theta_vec, score.get());
  }

  score_calc->AppendScore(batch, p_wt, args, score.get());

  return score;
}

}  // namespace core
}  // namespace artm
