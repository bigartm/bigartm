/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)

   This class proceeds Phi matrix regularization due to NetPLSA strategy.
   The formula of M-step is
   
   p_ut \propto n_ut + tau * \sum_{v \in U} w_uv n_t^2
                             ([p_vt / (|D_u| * |D_v|)] - [p_ut / (|D_u|^2)]),
   
   where U is the set of tokens of specific vertex modality of documents graph,
   w_uv is a weight of edge between vertices u and v,
   n_t is a normalized n_wt for tokens of U class_id (n_t = \sum_{v \in U} n_vt),
   D is an array of weights of vertices, len(D) == |U|
   
   The parameters of the regularizer:
   - topic_names (the names of topics to regularize, empty == all)
   - class_id (the name of specific class_id, should be equal to one in
               documents and is required)
   - transaction_typename (transaction typename to regularize,
                           if empty -> == DefaultTransactionTypeName)
   - weights w_uv (sparce matrix of weights w_uv, required)
   - weights D_u (vector with weights D_u, if is not set, all weights should be 1.0)
   - symmetric_weights (flag that indicates if w_uv is a symmetric sparce matrix or not)

   Note: max index of vertex in w_uv matrix should be less or equal to len(D) - 1.
*/

#pragma once

#include <string>
#include <vector>

#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

typedef std::unordered_map<int, std::unordered_map<int, float> > EdgeWeights;

class NetPlsaPhi : public RegularizerInterface {
 public:
  explicit NetPlsaPhi(const NetPlsaPhiConfig& config) : config_(config) {
    UpdateNetInfo(config);
  }

  virtual bool RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                             const ::artm::core::PhiMatrix& n_wt,
                             ::artm::core::PhiMatrix* result);

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();
  virtual google::protobuf::RepeatedPtrField<std::string> class_ids_to_regularize();

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  void UpdateNetInfo(const NetPlsaPhiConfig& config);

  NetPlsaPhiConfig config_;
  EdgeWeights edge_weights_;
  std::vector<std::string> vertex_name_;
};

}  // namespace regularizer
}  // namespace artm
