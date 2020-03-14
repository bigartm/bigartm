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

class ProcessorTransactionHelpers {
 public:
  static void TransactionInferThetaAndUpdateNwtSparse(
                                     const ProcessBatchesArgs& args,
                                     const Batch& batch,
                                     float batch_weight,
                                     const ::artm::core::PhiMatrix& p_wt,
                                     const RegularizeThetaAgentCollection& theta_agents,
                                     LocalThetaMatrix<float>* theta_matrix,
                                     NwtWriteAdapter* nwt_writer, util::Blas* blas,
                                     ThetaMatrix* new_cache_entry_ptr);

  ProcessorTransactionHelpers() = delete;
};

}  // namespace core
}  // namespace artm
