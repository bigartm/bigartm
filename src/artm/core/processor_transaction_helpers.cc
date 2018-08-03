// Copyright 2018, Additive Regularization of Topic Models.

#include "artm/core/processor_transaction_helpers.h"

namespace artm {
namespace core {

namespace {
const double kTransactionsEps = 1e-100;

struct IntVectorHasher {
  size_t operator()(const std::vector<int>& elems) const {
    size_t hash = 0;
    for (const int e : elems) {
      boost::hash_combine<std::string>(hash, std::to_string(e));
    }
    return hash;
  }
};

  typedef std::unordered_map<std::vector<int>, std::shared_ptr<TransactionInfo>, IntVectorHasher> TokenIdsToInfo;
}  // namespace

std::shared_ptr<BatchTransactionInfo> ProcessorTransactionHelpers::PrepareBatchInfo(
    const Batch& batch, const ProcessBatchesArgs& args, const ::artm::core::PhiMatrix& p_wt) {
  std::vector<float> n_dw_val;
  std::vector<int> n_dw_row_ptr;
  std::vector<int> n_dw_col_ind;

  std::unordered_map<Token, int, TokenHasher> token_to_index;
  TokenIdsToInfo token_ids_to_info;
  TransactionIdToInfo transaction_id_to_info;

  bool use_class_weight = false;
  std::unordered_map<ClassId, float> class_id_to_weight;
  if (args.class_id_size() > 0) {
    use_class_weight = true;
    for (int i = 0; i < args.class_id_size(); ++i) {
      class_id_to_weight.emplace(args.class_id(i), args.class_weight(i));
    }
  }

  bool use_transaction_weight = false;
  std::unordered_map<TransactionTypeName, float> tt_name_to_weight;
  if (args.transaction_typename_size() > 0) {
    use_transaction_weight = true;
    for (int i = 0; i < args.transaction_typename_size(); ++i) {
      tt_name_to_weight.emplace(args.transaction_typename(i), args.transaction_weight(i));
    }
  }

  // Weight of each transaction is a sum of weights of its tokens
  // (multiplied by their class_weight), multiplied by its transaction_weight
  std::vector<int> vec;
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
    const Item& item = batch.item(item_index);

    for (int t_index = 0; t_index < item.transaction_start_index_size() - 1; ++t_index) {
      const int start_index = item.transaction_start_index(t_index);
      const int end_index = item.transaction_start_index(t_index + 1);

      TransactionTypeName tt_name = batch.transaction_typename(item.transaction_typename_id(t_index));
      float tt_weight = 1.0f;
      if (use_transaction_weight) {
        auto iter = tt_name_to_weight.find(tt_name);
        tt_weight = (iter == tt_name_to_weight.end()) ? 0.0f : iter->second;
      }

      float transaction_weight = 0.0f;
      for (int idx = start_index; idx < end_index; ++idx) {
        const int token_id = item.token_id(idx);
        const float token_weight = item.token_weight(idx);

        ClassId class_id = batch.class_id(token_id);

        float class_weight = 1.0f;
        if (use_class_weight) {
          auto iter = class_id_to_weight.find(class_id);
          class_weight = (iter == class_id_to_weight.end()) ? 0.0f : iter->second;
        }

        transaction_weight += (token_weight * class_weight);
      }

      n_dw_val.push_back(transaction_weight * tt_weight);

      vec.clear();
      for (int i = start_index; i < end_index; ++i) {
        vec.push_back(item.token_id(i));
      }

      auto iter = token_ids_to_info.find(vec);
      if (iter != token_ids_to_info.end()) {
        n_dw_col_ind.push_back(iter->second->transaction_index);
      } else {
        std::vector<int> local_indices;
        std::vector<int> global_indices;

        for (int idx = start_index; idx < end_index; ++idx) {
          const int token_id = item.token_id(idx);
          auto token = Token(batch.class_id(token_id), batch.token(token_id));
          token_to_index.emplace(token, token_to_index.size());

          local_indices.push_back(token_to_index.size() - 1);
          global_indices.push_back(p_wt.token_index(token));
        }

        auto ptr = std::make_shared<TransactionInfo>(token_ids_to_info.size(), local_indices, global_indices);

        token_ids_to_info.emplace(vec, ptr);
        transaction_id_to_info.emplace(transaction_id_to_info.size(), ptr);

        n_dw_col_ind.push_back(token_ids_to_info.size() - 1);
      }
    }
  }
  n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));

  if (token_ids_to_info.size() != transaction_id_to_info.size()) {
    LOG(ERROR) << "Fatal error: token_ids_to_info.size() [ " << token_ids_to_info.size()
      << " ] != transaction_id_to_info.size() [ " << transaction_id_to_info.size();
  }

  return std::make_shared<BatchTransactionInfo>(
      std::make_shared<CsrMatrix<float>>(token_ids_to_info.size(),
                                         &n_dw_val, &n_dw_row_ptr, &n_dw_col_ind),
      transaction_id_to_info, token_to_index.size());
}

void ProcessorTransactionHelpers::TransactionInferThetaAndUpdateNwtSparse(
                                     const ProcessBatchesArgs& args,
                                     const Batch& batch,
                                     float batch_weight,
                                     std::shared_ptr<BatchTransactionInfo> batch_info,
                                     const ::artm::core::PhiMatrix& p_wt,
                                     const RegularizeThetaAgentCollection& theta_agents,
                                     LocalThetaMatrix<float>* theta_matrix,
                                     NwtWriteAdapter* nwt_writer, util::Blas* blas,
                                     ThetaMatrix* new_cache_entry_ptr) {
  if (!args.opt_for_avx()) {
    LOG(WARNING) << "Current version of BigARTM doesn't support 'opt_for_avx' == false"
      << " with complex transactions, option 'opt_for_avx' will be ignored";
  }

  LocalThetaMatrix<float> n_td(theta_matrix->num_topics(), theta_matrix->num_items());
  const int num_topics = p_wt.topic_size();
  const int docs_count = theta_matrix->num_items();
  const auto& sparse_ndx = *(batch_info->n_dx);

  LocalPhiMatrix<float> local_phi(batch_info->token_size, num_topics);
  LocalThetaMatrix<float> r_td(num_topics, 1);
  std::vector<float> helper_vector(num_topics, 0.0f);

  for (int d = 0; d < docs_count; ++d) {
    float* ntd_ptr = &n_td(0, d);
    float* theta_ptr = &(*theta_matrix)(0, d);  // NOLINT

    const int begin_index = sparse_ndx.row_ptr()[d];
    const int end_index = sparse_ndx.row_ptr()[d + 1];
    local_phi.InitializeZeros();
    bool item_has_tokens = false;
    for (int i = begin_index; i < end_index; ++i) {
      auto it = batch_info->transaction_id_to_info.find(sparse_ndx.col_ind()[i]);

      for (int k = 0; k < it->second->local_pwt_token_index.size(); ++k) {
        int global_index = it->second->global_pwt_token_index[k];
        if (global_index == ::artm::core::PhiMatrix::kUndefIndex) {
          continue;
        }

        item_has_tokens = true;

        float* local_phi_ptr = &local_phi(it->second->local_pwt_token_index[k], 0);
        p_wt.get(global_index, &helper_vector);
        for (int k = 0; k < num_topics; ++k) {
          local_phi_ptr[k] = helper_vector[k];
        }
      }
    }

    if (!item_has_tokens) {
      continue;  // continue to the next item
    }

    std::vector<float> p_xt_local(num_topics, 1.0f);
    for (int inner_iter = 0; inner_iter < args.num_document_passes(); ++inner_iter) {
      for (int k = 0; k < num_topics; ++k) {
        ntd_ptr[k] = 0.0f;
      }

      for (int i = begin_index; i < end_index; ++i) {
        std::fill(p_xt_local.begin(), p_xt_local.end(), 1.0f);
        auto it = batch_info->transaction_id_to_info.find(sparse_ndx.col_ind()[i]);

        for (int local_index : it->second->local_pwt_token_index) {
          const float* phi_ptr = &local_phi(local_index, 0);
          for (int k = 0; k < num_topics; ++k) {
            p_xt_local[k] *= phi_ptr[k];
          }
        }

        double p_dx_val = 0.0;
        for (int k = 0; k < num_topics; ++k) {
          p_dx_val += p_xt_local[k] * theta_ptr[k];
        }
        if (isZero(p_dx_val, kTransactionsEps)) {
          continue;
        }

        const double alpha = sparse_ndx.val()[i] / p_dx_val;
        for (int k = 0; k < num_topics; ++k) {
          ntd_ptr[k] += alpha * p_xt_local[k];
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

  CsrMatrix<float> sparse_nxd(sparse_ndx);
  sparse_nxd.Transpose(blas);

  std::vector<float> values(num_topics, 0.0f);
  std::vector<float> p_xt_local(num_topics, 1.0f);

  for (const auto& tuple : batch_info->transaction_id_to_info) {
    int transaction_index = tuple.first;

    std::fill(p_xt_local.begin(), p_xt_local.end(), 1.0f);
    for (int global_index : tuple.second->global_pwt_token_index) {
      if (global_index == ::artm::core::PhiMatrix::kUndefIndex) {
        continue;
      }
      p_wt.get(global_index, &helper_vector);
      for (int i = 0; i < num_topics; ++i) {
        p_xt_local[i] *= helper_vector[i];
      }
    }

    std::fill(helper_vector.begin(), helper_vector.end(), 0.0f);
    for (int i = sparse_nxd.row_ptr()[transaction_index]; i < sparse_nxd.row_ptr()[transaction_index + 1]; ++i) {
      int d = sparse_nxd.col_ind()[i];
      double p_xd_val = blas->sdot(num_topics, &p_xt_local[0], 1, &(*theta_matrix)(0, d), 1);  // NOLINT
      if (isZero(p_xd_val, kTransactionsEps)) {
        continue;
      }

      blas->saxpy(num_topics, sparse_nxd.val()[i] / p_xd_val,
        &(*theta_matrix)(0, d), 1, &helper_vector[0], 1);  // NOLINT
    }

    for (int global_index : tuple.second->global_pwt_token_index) {
      if (global_index == ::artm::core::PhiMatrix::kUndefIndex) {
        continue;
      }

      for (int topic_index = 0; topic_index < num_topics; ++topic_index) {
        values[topic_index] = p_xt_local[topic_index] * helper_vector[topic_index] * batch_weight;
      }
      nwt_writer->Store(global_index, values);
    }
  }
}

}  // namespace core
}  // namespace artm
