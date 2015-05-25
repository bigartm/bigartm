// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#ifndef SRC_ARTM_REGULARIZER_IMPROVE_COHERENCY_PHI_H_
#define SRC_ARTM_REGULARIZER_IMPROVE_COHERENCY_PHI_H_

#include <string>

#include "artm/messages.pb.h"
#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class ImproveCoherencyPhi : public RegularizerInterface {
 public:
  explicit ImproveCoherencyPhi(const ImproveCoherencyPhiConfig& config)
    : config_(config) {}

  virtual bool RegularizePhi(const ::artm::core::Regularizable& topic_model,
                             ::artm::core::TokenCollectionWeights* result);

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();
  virtual google::protobuf::RepeatedPtrField<std::string> class_ids_to_regularize();

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  ImproveCoherencyPhiConfig config_;
};

}  // namespace regularizer
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_IMPROVE_COHERENCY_PHI_H_
