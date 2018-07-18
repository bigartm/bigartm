/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)

   This class proceeds Phi matrix smoothing or sparsing.
   The formula of M-step is
   
   p_wt \propto n_wt + tau * f(p_wt) * dict[w],
   
   where f is a transform function, which is p_wt multiplied on
   the derivative of function under KL-divergence and dict[w]
   is a token_value from dictionary if it was provided, or 1.
   Note that dictionary usage will set to zero each token, that
   hasn't token_value in it.
   
   The parameters of the regularizer:
   - topic_names (the names of topics to regularize, empty == all)
   - class_ids (class ids to regularize, empty == all)
   - transaction_typenames (transaction typenames to regularize, empty == all)
   - dictionary_name
   - transform_function (default is 1, corresponds log() under
     KL-divergence)
*/

#pragma once

#include <memory>
#include <string>

#include "artm/regularizer_interface.h"
#include "artm/core/transform_function.h"

namespace artm {
namespace regularizer {

class SmoothSparsePhi : public RegularizerInterface {
 public:
  explicit SmoothSparsePhi(const SmoothSparsePhiConfig& config);

  virtual bool RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                             const ::artm::core::PhiMatrix& n_wt,
                             ::artm::core::PhiMatrix* result);

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();
  virtual google::protobuf::RepeatedPtrField<std::string> class_ids_to_regularize();

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  SmoothSparsePhiConfig config_;
  std::shared_ptr<artm::core::TransformFunction> transform_function_;
};

}  // namespace regularizer
}  // namespace artm
