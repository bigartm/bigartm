// Copyright 2018, Additive Regularization of Topic Models.

#include "artm/core/processor_transaction_helpers.h"

namespace artm {
namespace core {

namespace {
const double kTransactionsEps = 1e-100;
}

void ProcessorTransactionHelpers::TransactionInferThetaAndUpdateNwtSparse(
                                     const ProcessBatchesArgs& args,
                                     const Batch& batch,
                                     float batch_weight,
                                     const ::artm::core::PhiMatrix& p_wt,
                                     const RegularizeThetaAgentCollection& theta_agents,
                                     LocalThetaMatrix<float>* theta_matrix,
                                     NwtWriteAdapter* nwt_writer, util::Blas* blas,
                                     ThetaMatrix* new_cache_entry_ptr) {
  if (!args.opt_for_avx()) {
    LOG(WARNING) << "Current version of BigARTM doesn't support 'opt_for_avx' == false"
      << " with complex transactions, option 'opt_for_avx' will be ignored";
  }

  const int num_topics = p_wt.topic_size();
  const int docs_count = theta_matrix->num_items();

  LocalThetaMatrix<float> n_td(num_topics, docs_count);
  LocalThetaMatrix<float> r_td(num_topics, 1);

  bool use_transaction_weight = false;
  std::unordered_map<TransactionTypeName, float> tt_name_to_weight;
  if (args.transaction_typename_size() > 0) {
    use_transaction_weight = true;
    for (int i = 0; i < args.transaction_typename_size(); ++i) {
      tt_name_to_weight.emplace(args.transaction_typename(i), args.transaction_weight(i));
    }
  }

  bool use_class_weight = false;
  std::unordered_map<ClassId, float> class_id_to_weight;
  if (args.class_weight_size() > 0) {
    use_class_weight = true;
    for (int i = 0; i < args.class_weight_size(); ++i) {
      class_id_to_weight.emplace(args.class_id(i), args.class_weight(i));
    }
  }

  for (int d = 0; d < docs_count; ++d) {
    float *ntd_ptr = &n_td(0, d);
    float *theta_ptr = &(*theta_matrix)(0, d);  // NOLINT

    const auto &item = batch.item(d);

    for (int inner_iter = 0; inner_iter < args.num_document_passes(); ++inner_iter) {
      for (int k = 0; k < num_topics; ++k) {
        ntd_ptr[k] = 0.0f;
      }

      for (int t_index = 0; t_index < item.transaction_start_index_size() - 1; ++t_index) {
        const int start_index = item.transaction_start_index(t_index);
        const int end_index = item.transaction_start_index(t_index + 1);

        const TransactionTypeName &tt_name = batch.transaction_typename(item.transaction_typename_id(t_index));
        float tt_weight = 1.0f;
        if (use_transaction_weight) {
          auto iter = tt_name_to_weight.find(tt_name);
          tt_weight = (iter == tt_name_to_weight.end()) ? 0.0f : iter->second;
        }

        double p_dx_val = 0.0;
        for (int k = 0; k < num_topics; ++k) {
          double pre_p_dx_val = theta_ptr[k];
          for (int token_id = start_index; token_id < end_index; ++token_id) {
            pre_p_dx_val *= p_wt.get(p_wt.token_index({batch.class_id(item.token_id(token_id)),
                                                       batch.token(item.token_id(token_id))}), k);
          }
          p_dx_val += pre_p_dx_val;
        }

        if (isZero(p_dx_val, kTransactionsEps)) {
          continue;
        }

        for (int k = 0; k < num_topics; ++k) {
          double pre_p_dx_val = 1.0;
          for (int token_id = start_index; token_id < end_index; ++token_id) {
            pre_p_dx_val *= p_wt.get(p_wt.token_index({batch.class_id(item.token_id(token_id)),
                                                       batch.token(item.token_id(token_id))}), k);
          }

          ntd_ptr[k] += (tt_weight * pre_p_dx_val / p_dx_val);
        }
      }

      for (int k = 0; k < num_topics; ++k) {
        theta_ptr[k] *= ntd_ptr[k];
      }

      r_td.InitializeZeros();
      theta_agents.Apply(d, inner_iter, num_topics, theta_ptr, r_td.get_data());
    }
  }

  ProcessorHelpers::CreateThetaCacheEntry(new_cache_entry_ptr, theta_matrix, batch, p_wt, args);

  if (nwt_writer == nullptr) {
    return;
  }

  for (int d = 0; d < docs_count; ++d) {
    const auto &item = batch.item(d);
    for (int current_token_id = 0; current_token_id < p_wt.token_size(); ++current_token_id) {
      const auto &current_token = p_wt.token(current_token_id);

      float class_weight = 1.0f;
      if (use_class_weight) {
        auto iter = class_id_to_weight.find(current_token.class_id);
        class_weight = (iter == class_id_to_weight.end()) ? 0.0f : iter->second;
      }

      for (int t_index = 0; t_index < item.transaction_start_index_size() - 1; ++t_index) {
        const int start_index = item.transaction_start_index(t_index);
        const int end_index = item.transaction_start_index(t_index + 1);

        double p_dx_val = 0.0;

        const TransactionTypeName &tt_name = batch.transaction_typename(item.transaction_typename_id(t_index));
        float tt_weight = 1.0f;
        if (use_transaction_weight) {
          auto iter = tt_name_to_weight.find(tt_name);
          tt_weight = (iter == tt_name_to_weight.end()) ? 0.0f : iter->second;
        }

        bool transaction_contains_token = false;
        for (int token_id = start_index; token_id < end_index; ++token_id) {
          if (batch.token(item.token_id(token_id)) == current_token.keyword &&
              batch.class_id(item.token_id(token_id)) == current_token.class_id) {
            transaction_contains_token = true;
            break;
          }
        }

        if (!transaction_contains_token) {
          continue;
        }

        for (int k = 0; k < num_topics; ++k) {
          double pre_p_dx_val = (*theta_matrix)(k, d);
          for (int token_id = start_index; token_id < end_index; ++token_id) {
            pre_p_dx_val *= p_wt.get(p_wt.token_index({batch.class_id(item.token_id(token_id)),
                                                       batch.token(item.token_id(token_id))}), k);
          }

          p_dx_val += pre_p_dx_val;
        }

        std::vector<float> values(num_topics, 0.0f);
        for (int k = 0; k < num_topics; ++k) {
          double value = (*theta_matrix)(k, d);
          for (int token_id = start_index; token_id < end_index; ++token_id) {
            value *= p_wt.get(p_wt.token_index({batch.class_id(item.token_id(token_id)),
                                                batch.token(item.token_id(token_id))}), k);
          }

          values[k] = (tt_weight * class_weight) * value * batch_weight / p_dx_val;
        }

        nwt_writer->Store(current_token_id, values);
      }
    }
  }
}



}  // namespace core
}  // namespace artm
