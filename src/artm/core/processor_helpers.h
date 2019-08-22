// Copyright 2018, Additive Regularization of Topic Models.

#pragma once

#include <unordered_map>
#include <memory>
#include <vector>
#include <string>

#include "artm/core/phi_matrix.h"
#include "artm/core/phi_matrix_operations.h"
#include "artm/core/instance.h"
#include "artm/core/helpers.h"
#include "artm/core/protobuf_helpers.h"
#include "artm/core/score_manager.h"

#include "artm/regularizer_interface.h"
#include "artm/score_calculator_interface.h"

#include "artm/utility/blas.h"

namespace util = artm::utility;
using ::util::CsrMatrix;
using ::util::LocalThetaMatrix;
using ::util::LocalPhiMatrix;

const float kProcessorEps = 1e-16f;

namespace artm {
namespace core {

class RegularizeThetaAgentCollection : public RegularizeThetaAgent {
 private:
  std::vector<std::shared_ptr<RegularizeThetaAgent>> agents_;

 public:
  void AddAgent(std::shared_ptr<RegularizeThetaAgent> agent) {
    if (agent != nullptr) {
      agents_.push_back(agent);
    }
  }

  bool empty() const { return agents_.empty(); }

  virtual void Apply(int item_index, int inner_iter, int topics_size, const float* n_td, float* r_td) const {
    for (auto& agent : agents_) {
      agent->Apply(item_index, inner_iter, topics_size, n_td, r_td);
    }
  }

  virtual void Apply(int inner_iter, const LocalThetaMatrix<float>& n_td, LocalThetaMatrix<float>* r_td) const {
    for (auto& agent : agents_) {
      agent->Apply(inner_iter, n_td, r_td);
    }
  }
};

class RegularizePtdwAgentCollection : public RegularizePtdwAgent {
 private:
  std::vector<std::shared_ptr<RegularizePtdwAgent>> agents_;

 public:
  void AddAgent(std::shared_ptr<RegularizePtdwAgent> agent) {
    if (agent != nullptr) {
      agents_.push_back(agent);
    }
  }

  bool empty() const { return agents_.empty(); }

  virtual void Apply(int item_index, int inner_iter, LocalPhiMatrix<float>* ptdw) const {
    for (auto& agent : agents_) {
      agent->Apply(item_index, inner_iter, ptdw);
    }
  }
};

class NormalizeThetaAgent : public RegularizeThetaAgent {
 public:
  virtual void Apply(int item_index, int inner_iter, int topics_size, const float* n_td, float* r_td) const {
    float sum = 0.0f;
    for (int topic_index = 0; topic_index < topics_size; ++topic_index) {
      float val = n_td[topic_index] + r_td[topic_index];
      if (val > 0) {
        sum += val;
      }
    }

    float sum_inv = sum > 0.0f ? (1.0f / sum) : 0.0f;
    for (int topic_index = 0; topic_index < topics_size; ++topic_index) {
      float val = sum_inv * (n_td[topic_index] + r_td[topic_index]);
      if (val < kProcessorEps) {
        val = 0.0f;
      }

      // Hack-hack, write normalized values back to n_td
      const_cast<float*>(n_td)[topic_index] = val;
    }
  }
};

class NwtWriteAdapter {
 public:
  explicit NwtWriteAdapter(PhiMatrix* n_wt) : n_wt_(n_wt) { }

  void Store(int nwt_token_id, const std::vector<float>& nwt_vector) {
    assert(nwt_vector.size() == n_wt_->topic_size());
    assert((nwt_token_id >= 0) && (nwt_token_id < n_wt_->token_size()));
    n_wt_->increase(nwt_token_id, nwt_vector);
  }

  PhiMatrix* n_wt() {
    return n_wt_;
  }

 private:
  PhiMatrix* n_wt_;
};

class ProcessorHelpers {
 public:
  static void CreateThetaCacheEntry(ThetaMatrix* new_cache_entry_ptr,
                                    LocalThetaMatrix<float>* theta_matrix,
                                    const Batch& batch,
                                    const PhiMatrix& p_wt,
                                    const ProcessBatchesArgs& args);

  static void CreatePtdwCacheEntry(ThetaMatrix* new_cache_entry_ptr,
                                   LocalPhiMatrix<float>* ptdw_matrix,
                                   const Batch& batch,
                                   int item_index,
                                   int topic_size);

  static std::shared_ptr<LocalThetaMatrix<float>> InitializeTheta(int topic_size,
                                                                  const Batch& batch,
                                                                  const ProcessBatchesArgs& args,
                                                                  const ThetaMatrix* cache);

  static std::shared_ptr<LocalPhiMatrix<float>> InitializePhi(const Batch& batch,
                                                              const ::artm::core::PhiMatrix& p_wt);

  static void CreateRegularizerAgents(const Batch& batch,
                                      const ProcessBatchesArgs& args,
                                      Instance* instance,
                                      RegularizeThetaAgentCollection* theta_agents,
                                      RegularizePtdwAgentCollection* ptdw_agents);

  static std::shared_ptr<CsrMatrix<float>> InitializeSparseNdw(const Batch& batch,
                                                               const ProcessBatchesArgs& args);

  static void FindBatchTokenIds(const Batch& batch,
                                const PhiMatrix& phi_matrix,
                                std::vector<int>* token_id);

  static std::shared_ptr<Score> CalcScores(ScoreCalculatorInterface* score_calc,
                                           const Batch& batch,
                                           const PhiMatrix& p_wt,
                                           const ProcessBatchesArgs& args,
                                           const LocalThetaMatrix<float>& theta_matrix);

  static void InferPtdwAndUpdateNwtSparse(const ProcessBatchesArgs& args,
                                          const Batch& batch,
                                          float batch_weight,
                                          const CsrMatrix<float>& sparse_ndw,
                                          const ::artm::core::PhiMatrix& p_wt,
                                          const RegularizeThetaAgentCollection& theta_agents,
                                          const RegularizePtdwAgentCollection& ptdw_agents,
                                          LocalThetaMatrix<float>* theta_matrix,
                                          NwtWriteAdapter* nwt_writer, util::Blas* blas,
                                          ThetaMatrix* new_cache_entry_ptr = nullptr,
                                          ThetaMatrix* new_ptdw_cache_entry_ptr = nullptr);

  static void InferThetaAndUpdateNwtSparse(const ProcessBatchesArgs& args,
                                           const Batch& batch,
                                           float batch_weight,
                                           const CsrMatrix<float>& sparse_ndw,
                                           const ::artm::core::PhiMatrix& p_wt,
                                           const RegularizeThetaAgentCollection& theta_agents,
                                           LocalThetaMatrix<float>* theta_matrix,
                                           NwtWriteAdapter* nwt_writer,
                                           util::Blas* blas,
                                           ThetaMatrix* new_cache_entry_ptr = nullptr);

  ProcessorHelpers() = delete;
};

}  // namespace core
}  // namespace artm
