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

struct TransactionInfo {
  int transaction_index;
  std::vector<int> local_pwt_token_index;
  std::vector<int> global_pwt_token_index;

  TransactionInfo(int _transaction_index,
                  const std::vector<int>& _local_pwt_token_index,
                  const std::vector<int>& _global_pwt_token_index)
      : transaction_index(_transaction_index)
      , local_pwt_token_index(_local_pwt_token_index)
      , global_pwt_token_index(_global_pwt_token_index) { }
};

typedef std::unordered_map<std::vector<int>, std::shared_ptr<TransactionInfo>, IntVectorHasher> TokenIdsToInfo;
typedef std::unordered_map<int, std::shared_ptr<TransactionInfo>> TransactionIdToInfo;

struct BatchTransactionInfo {
  TokenIdsToInfo token_ids_to_info;
  TransactionIdToInfo transaction_id_to_info;
  int token_size;

  BatchTransactionInfo(const TokenIdsToInfo& _token_ids_to_info,
                       const TransactionIdToInfo& _transaction_id_to_info,
                       int _token_size)
      : token_ids_to_info(_token_ids_to_info)
      , transaction_id_to_info(_transaction_id_to_info)
      , token_size(_token_size) { }
};

class ProcessorTransactionHelpers {
 public:
  static std::shared_ptr<BatchTransactionInfo> GetBatchTransactionsInfo(
      const Batch& batch, const ::artm::core::PhiMatrix& p_wt);

  static std::shared_ptr<CsrMatrix<float>> InitializeSparseNdx(
    const Batch& batch, const ProcessBatchesArgs& args, const TokenIdsToInfo& transaction_info);

  static void TransactionInferThetaAndUpdateNwtSparse(
                                     const ProcessBatchesArgs& args,
                                     const Batch& batch,
                                     float batch_weight,
                                     const CsrMatrix<float>& sparse_ndx,
                                     const TransactionIdToInfo& transaction_id_to_info,
                                     int token_size,
                                     const ::artm::core::PhiMatrix& p_wt,
                                     const RegularizeThetaAgentCollection& theta_agents,
                                     LocalThetaMatrix<float>* theta_matrix,
                                     NwtWriteAdapter* nwt_writer, util::Blas* blas,
                                     ThetaMatrix* new_cache_entry_ptr);

  ProcessorTransactionHelpers() = delete;
};

}  // namespace core
}  // namespace artm
