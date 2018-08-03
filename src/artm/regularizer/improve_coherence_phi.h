/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)

   This class proceeds improoving the coherency of topics in Phi matrix.
   The formula of M-step is
   
   p_wt \propto n_wt + tau * \sum_{v in W} CoocDict_{wv} n_{vt}.
   
   CoocDict is a dictionary with information about pairwise tokens
   cooccurrence, that is using in coherence score. Note that tokens
   without such information will be skipped.
   
   The parameters of the regularizer:
   - topic_names (the names of topics to regularize, empty == all)
   - class_ids (class ids to regularize, empty == all)
   - transaction_typenames (transaction typenames to regularize, empty == all)
   - dictionary_name (strongly required parameter)
*/

#pragma once

#include <string>

#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class ImproveCoherencePhi : public RegularizerInterface {
 public:
  explicit ImproveCoherencePhi(const ImproveCoherencePhiConfig& config)
    : config_(config) { }

  virtual bool RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                             const ::artm::core::PhiMatrix& n_wt,
                             ::artm::core::PhiMatrix* result);

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();
  virtual google::protobuf::RepeatedPtrField<std::string> class_ids_to_regularize();

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  ImproveCoherencePhiConfig config_;
};

}  // namespace regularizer
}  // namespace artm
