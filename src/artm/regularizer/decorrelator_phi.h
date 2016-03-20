/* Copyright 2014, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)

   This class proceeds the decorrelation of topics in Phi matrix.
   The formula of M-step is
   
   p_wt \propto n_wt - tau * p_wt * \sum_{s \in T\t} p_ws.
   
   The parameters of the regularizer:
   - topic_names (the names of topics to regularize, empty == all)
   - class_ids (class ids to regularize, empty == all)

*/

#ifndef SRC_ARTM_REGULARIZER_DECORRELATOR_PHI_H_
#define SRC_ARTM_REGULARIZER_DECORRELATOR_PHI_H_

#include <string>

#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class DecorrelatorPhi : public RegularizerInterface {
 public:
  explicit DecorrelatorPhi(const DecorrelatorPhiConfig& config)
    : config_(config) {}

  virtual bool RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                             const ::artm::core::PhiMatrix& n_wt,
                             ::artm::core::PhiMatrix* result);

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();
  virtual google::protobuf::RepeatedPtrField<std::string> class_ids_to_regularize();

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  DecorrelatorPhiConfig config_;
};

}  // namespace regularizer
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_DECORRELATOR_PHI_H_
