// Copyright 2018, Additive Regularization of Topic Models.

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "artm/core/phi_matrix.h"
#include "artm/core/phi_matrix_operations.h"
#include "artm/core/instance.h"
#include "artm/core/processor_helpers.h"

#include "artm/utility/blas.h"

namespace util = artm::utility;
using ::util::CsrMatrix;
using ::util::LocalThetaMatrix;
using ::util::LocalPhiMatrix;

namespace artm {
namespace core {

struct IntVectorHasher {
  size_t operator()(const std::vector<int>& elems) const {
    size_t hash = 0;
    for (const int e : elems) {
      boost::hash_combine<std::string>(hash, std::to_string(e));
    }
    return hash;
  }
};

struct TokenVectorHasher {
  size_t operator()(const std::vector<Token>& elems) const {
    size_t hash = 0;
    for (const auto& e : elems) {
      boost::hash_combine<size_t>(hash, e.hash());
    }
    return hash;
  }
};

typedef std::unordered_map<ClassId, std::unordered_set<TransactionType, TransactionHasher>> ClassIdToTt;
typedef std::unordered_map<std::vector<Token>, int, TokenVectorHasher> TransactionToIndex;

struct BatchTransactionInfo {
  ClassIdToTt class_id_to_tt;
  std::unordered_map<std::vector<int>, int, IntVectorHasher> transaction_ids_to_index;
  TransactionToIndex transaction_to_index;
  std::vector<std::vector<Token>> transactions;
  std::unordered_map<Token, int, TokenHasher> token_to_index;

  BatchTransactionInfo(const ClassIdToTt& _class_id_to_tt,
                       const std::unordered_map<std::vector<int>, int, IntVectorHasher>& _transaction_ids_to_index,
                       const TransactionToIndex& _transaction_to_index,
                       const std::vector<std::vector<Token>>& _transactions,
                       const std::unordered_map<Token, int, TokenHasher>& _token_to_index)
      : class_id_to_tt(_class_id_to_tt)
      , transaction_ids_to_index(_transaction_ids_to_index)
      , transaction_to_index(_transaction_to_index)
      , transactions(_transactions)
      , token_to_index(_token_to_index){ }
};

class ProcessorTransactionHelpers {
 public:
  static std::shared_ptr<BatchTransactionInfo> GetBatchTransactionsInfo(const Batch& batch);

  static std::shared_ptr<CsrMatrix<float>> InitializeSparseNdx(const Batch& batch,
      const ProcessBatchesArgs& args, const ClassIdToTt& class_id_to_tt,
      const std::unordered_map<std::vector<int>, int, IntVectorHasher>& transaction_to_index);

  static void TransactionInferThetaAndUpdateNwtSparse(
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
                                     ThetaMatrix* new_cache_entry_ptr);

  ProcessorTransactionHelpers() = delete;
};

}  // namespace core
}  // namespace artm
