// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#ifndef SRC_ARTM_REGULARIZER_DECORRELATOR_PHI_H_
#define SRC_ARTM_REGULARIZER_DECORRELATOR_PHI_H_

#include <string>

#include "artm/messages.pb.h"
#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class DecorrelatorPhi : public RegularizerInterface {
 public:
  explicit DecorrelatorPhi(const DecorrelatorPhiConfig& config)
    : config_(config) {}

  virtual bool RegularizePhi(const ::artm::core::Regularizable& topic_model,
                             ::artm::core::TokenCollectionWeights* result);

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();
  virtual google::protobuf::RepeatedPtrField<std::string> class_ids_to_regularize();

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  DecorrelatorPhiConfig config_;
};

}  // namespace regularizer
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_DECORRELATOR_PHI_H_
