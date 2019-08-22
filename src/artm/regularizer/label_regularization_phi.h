/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)

   This class proceeds Phi matrix Label Regularization.
   The formula of M-step is
   
   p_wt \propto n_wt + tau * dict[w] * \frac{p_wt * n_t}{\sum_{s \in T} p_ws n_s},
   
   where dict[w] is a token_value from dictionary if it was provided, or 1.
   Note that dictionary usage will set to zero each token, that
   hasn't token_value in it. token_value should contain the value of
   empirical frequencies of tokens in collection. This regularizer is mostly
   using for <class-topic> matrix in classification topic models.
   
   The parameters of the regularizer:
   - topic_names (the names of topics to regularize, empty == all)
   - class_ids (class ids to regularize, empty == all)
   - transaction_typenames (transaction typenames to regularize, empty == all)
   - dictionary_name

*/

#pragma once

#include <string>

#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class LabelRegularizationPhi : public RegularizerInterface {
 public:
  explicit LabelRegularizationPhi(const LabelRegularizationPhiConfig& config) : config_(config) { }

  virtual bool RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                             const ::artm::core::PhiMatrix& n_wt,
                             ::artm::core::PhiMatrix* result);

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();
  virtual google::protobuf::RepeatedPtrField<std::string> class_ids_to_regularize();

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  LabelRegularizationPhiConfig config_;
};

}  // namespace regularizer
}  // namespace artm
