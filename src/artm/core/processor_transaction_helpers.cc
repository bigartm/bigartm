// Copyright 2018, Additive Regularization of Topic Models.

#include "artm/core/processor_transaction_helpers.h"

namespace artm {
namespace core {

std::shared_ptr<BatchTransactionInfo> ProcessorTransactionHelpers::GetBatchTransactionsInfo(const Batch& batch) {
  ClassIdToTt class_id_to_tt;
  std::unordered_map<std::vector<int>, int, IntVectorHasher> transaction_ids_to_index;
  TransactionToIndex transaction_to_index;
  std::vector<std::vector<Token>> transactions;
  std::unordered_map<Token, int, TokenHasher> token_to_index;

  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    const Item& item = batch.item(item_index);

    for (int token_index = 0; token_index < item.transaction_start_index_size(); ++token_index) {
      const int start_index = item.transaction_start_index(token_index);
      const int end_index = (token_index + 1) < item.transaction_start_index_size() ?
                            item.transaction_start_index(token_index + 1) :
                            item.transaction_token_id_size();
      std::vector<int> vec;
      for (int i = start_index; i < end_index; ++i) {
        vec.push_back(item.transaction_token_id(i));
      }
      auto iter = transaction_ids_to_index.find(vec);
      if (iter == transaction_ids_to_index.end()) {
        transaction_ids_to_index.insert(std::make_pair(vec, transaction_ids_to_index.size()));

        std::string str;
        for (int token_id = start_index; token_id < end_index; ++token_id) {
          auto& tmp = batch.class_id(item.transaction_token_id(token_id));
          str += (token_id == start_index) ? tmp : TransactionSeparator + tmp;
        }

        TransactionType tt(str);
        for (const ClassId& class_id : tt.AsVector()) {
          class_id_to_tt[class_id].emplace(tt);
        }

        std::vector<Token> transaction;
        for (int idx = start_index; idx < end_index; ++idx) {
          const int token_id = item.transaction_token_id(idx);
          auto token = Token(batch.class_id(token_id), batch.token(token_id), tt);
          transaction.push_back(token);
          token_to_index.insert(std::make_pair(token, token_to_index.size()));
        }
        transactions.push_back(transaction);
        transaction_to_index.insert(std::make_pair(transaction, transaction_to_index.size()));
      }
    }
  }

  if (transaction_to_index.size() != transactions.size()) {
    LOG(ERROR) << "Fatal error: transaction_to_index.size() [ " << transaction_to_index.size()
      << " ] != transactions.size() [ " << transactions.size();
  }

  if (transaction_ids_to_index.size() != transactions.size()) {
    LOG(ERROR) << "Fatal error: transaction_ids_to_index.size() [ " << transaction_ids_to_index.size()
               << " ] != transactions.size() [ " << transactions.size();
  }

  return std::make_shared<BatchTransactionInfo>(
    BatchTransactionInfo(class_id_to_tt, transaction_ids_to_index,
                         transaction_to_index, transactions, token_to_index));
}

std::shared_ptr<CsrMatrix<float>> ProcessorTransactionHelpers::InitializeSparseNdx(const Batch& batch,
    const ProcessBatchesArgs& args, const ClassIdToTt& class_id_to_tt,
    const std::unordered_map<std::vector<int>, int, IntVectorHasher>& transaction_to_index) {
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

  int max_doc_len = 0;
  std::vector<int> vec;
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
    const Item& item = batch.item(item_index);

    for (int token_index = 0; token_index < item.transaction_start_index_size(); ++token_index) {
      const int start_index = item.transaction_start_index(token_index);
      const int end_index = (token_index + 1) < item.transaction_start_index_size() ?
                            item.transaction_start_index(token_index + 1) :
                            item.transaction_token_id_size();

      float transaction_weight = 1.0f;
      if (use_weights) {
        std::string str;
        for (int token_id = start_index; token_id < end_index; ++token_id) {
          auto& tmp = batch.class_id(item.transaction_token_id(token_id));
          str += (token_id == start_index) ? tmp : TransactionSeparator + tmp;
        }
        auto iter = tt_to_weight.find(TransactionType(str));
        transaction_weight = (iter == tt_to_weight.end()) ? 0.0f : iter->second;
      }

      const float token_weight = item.token_weight(token_index);
      n_dw_val.push_back(transaction_weight * token_weight);

      vec.clear();
      for (int i = start_index; i < end_index; ++i) {
        vec.push_back(item.transaction_token_id(i));
      }
      auto iter = transaction_to_index.find(vec);
      if (iter != transaction_to_index.end()) {
        n_dw_col_ind.push_back(iter->second);
      } else {
        std::stringstream ss;
        ss << "Fatal error: transaction_to_index doesn't contain transaction from indices:";
        for (const int e : vec) {
          ss << " " << e;
        }
        ss << " read from item with index " << item_index << " from batch " << batch.id()
           << ", empty matrix will be returned for this batch.";
        LOG(ERROR) << ss.str();

        return std::make_shared<CsrMatrix<float>>(0, 0, 0);
      }
    }
  }
  n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));

  return std::make_shared<CsrMatrix<float>>(
    transaction_to_index.size(), &n_dw_val, &n_dw_row_ptr, &n_dw_col_ind);
}

void ProcessorTransactionHelpers::TransactionInferThetaAndUpdateNwtSparse(
                                     const ProcessBatchesArgs& args,
                                     const Batch& batch,
                                     float batch_weight,
                                     const CsrMatrix<float>& sparse_ndx,
                                     const TransactionToIndex& transaction_to_index,
                                     const std::unordered_map<Token, int, TokenHasher>& token_to_local_index,
                                     const std::vector<std::vector<Token>>& transactions,
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

  LocalPhiMatrix<float> local_phi(token_to_local_index.size(), num_topics);
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
      int w = sparse_ndx.col_ind()[i];
      auto& transaction = transactions[w];
      for (const auto& token : transaction) {
        if (p_wt.token_index(token) == ::artm::core::PhiMatrix::kUndefIndex) {
          continue;
        }
        auto iter = token_to_local_index.find(token);
        if (iter == token_to_local_index.end()) {
          continue;
        }
        item_has_tokens = true;
        float* local_phi_ptr = &local_phi(iter->second, 0);
        p_wt.get(p_wt.token_index(token), &helper_vector);
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
        int w = sparse_ndx.col_ind()[i];
        std::fill(p_xt_local.begin(), p_xt_local.end(), 1.0f);
        auto& transaction = transactions[w];
        for (const auto& token : transaction) {
          auto iter = token_to_local_index.find(token);

          if (iter == token_to_local_index.end()) {
            continue;
          }

          const float* phi_ptr = &local_phi(iter->second, 0);
          for (int k = 0; k < num_topics; ++k) {
            p_xt_local[k] *= phi_ptr[k];
          }
        }

        float p_dx_val = 0;
        for (int k = 0; k < num_topics; ++k) {
          p_dx_val += p_xt_local[k] * theta_ptr[k];
        }
        if (p_dx_val == 0) {
          continue;
        }

        const float alpha = sparse_ndx.val()[i] / p_dx_val;
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
  for (const auto& transaction : transactions) {
    auto tr_iter = transaction_to_index.find(transaction);
    if (tr_iter == transaction_to_index.end()) {
      continue;
    }
    int transaction_index = tr_iter->second;

    std::fill(p_xt_local.begin(), p_xt_local.end(), 1.0f);
    for (const auto& token : transaction) {
      int phi_token_index = p_wt.token_index(token);
      if (phi_token_index == ::artm::core::PhiMatrix::kUndefIndex) {
        continue;
      }

      p_wt.get(phi_token_index, &helper_vector);
      for (int i = 0; i < num_topics; ++i) {
        p_xt_local[i] *= helper_vector[i];
      }
    }

    std::fill(helper_vector.begin(), helper_vector.end(), 0.0f);
    for (int i = sparse_nxd.row_ptr()[transaction_index]; i < sparse_nxd.row_ptr()[transaction_index + 1]; ++i) {
      int d = sparse_nxd.col_ind()[i];
      float p_xd_val = blas->sdot(num_topics, &p_xt_local[0], 1, &(*theta_matrix)(0, d), 1);  // NOLINT
      if (p_xd_val == 0) {
        continue;
      }

      blas->saxpy(num_topics, sparse_nxd.val()[i] / p_xd_val,
        &(*theta_matrix)(0, d), 1, &helper_vector[0], 1);  // NOLINT
    }

    for (const auto& token : transaction) {
      int phi_token_index = p_wt.token_index(token);
      if (phi_token_index == ::artm::core::PhiMatrix::kUndefIndex) {
        continue;
      }

      for (int topic_index = 0; topic_index < num_topics; ++topic_index) {
        values[topic_index] = p_xt_local[topic_index] * helper_vector[topic_index] * batch_weight;
      }

      nwt_writer->Store(-1, phi_token_index, values);
    }
  }
}

}  // namespace core
}  // namespace artm
