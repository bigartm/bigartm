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

typedef std::unordered_map<int, std::shared_ptr<TransactionInfo>> TransactionIdToInfo;

struct BatchTransactionInfo {
  std::shared_ptr<CsrMatrix<float>> n_dx;
  TransactionIdToInfo transaction_id_to_info;
  int token_size;

  BatchTransactionInfo(std::shared_ptr<CsrMatrix<float>> _n_dx,
                       const TransactionIdToInfo& _transaction_id_to_info,
                       int _token_size)
      : n_dx(_n_dx), transaction_id_to_info(_transaction_id_to_info), token_size(_token_size) { }
};

class ProcessorTransactionHelpers {
 public:
  static std::shared_ptr<BatchTransactionInfo> PrepareBatchInfo(
    const Batch& batch, const ProcessBatchesArgs& args, const ::artm::core::PhiMatrix& p_wt);

  static void TransactionInferThetaAndUpdateNwtSparse(
                                     const ProcessBatchesArgs& args,
                                     const Batch& batch,
                                     float batch_weight,
                                     std::shared_ptr<BatchTransactionInfo> batch_info,
                                     const ::artm::core::PhiMatrix& p_wt,
                                     const RegularizeThetaAgentCollection& theta_agents,
                                     LocalThetaMatrix<float>* theta_matrix,
                                     NwtWriteAdapter* nwt_writer, util::Blas* blas,
                                     ThetaMatrix* new_cache_entry_ptr);

  ProcessorTransactionHelpers() = delete;
};

}  // namespace core
}  // namespace artm
