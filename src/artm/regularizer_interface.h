// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#ifndef SRC_ARTM_REGULARIZER_INTERFACE_H_
#define SRC_ARTM_REGULARIZER_INTERFACE_H_

#include <vector>
#include <map>
#include <memory>
#include <string>

#include "artm/messages.pb.h"
#include "artm/utility/blas.h"
#include "artm/core/common.h"
#include "artm/core/dictionary.h"
#include "artm/core/exceptions.h"

#include "glog/logging.h"

namespace artm {

namespace core {
  // Forward declarations
class Dictionary;
class PhiMatrix;
template<typename K, typename T> class ThreadSafeCollectionHolder;
typedef ThreadSafeCollectionHolder<std::string, Dictionary> ThreadSafeDictionaryCollection;
}

class RegularizeThetaAgent {
 public:
  virtual ~RegularizeThetaAgent() {}
  virtual void Apply(int item_index, int inner_iter, int topics_size, float* theta) const = 0;
};

class RegularizePtdwAgent {
 public:
  virtual ~RegularizePtdwAgent() {}
  virtual void Apply(int item_index, int inner_iter, ::artm::utility::DenseMatrix<float>* ptdw) const = 0;
};

class RegularizerInterface {
 public:
  RegularizerInterface() {}
  virtual ~RegularizerInterface() { }

  virtual std::shared_ptr<RegularizeThetaAgent>
  CreateRegularizeThetaAgent(const Batch& batch, const ProcessBatchesArgs& args, double tau) {
    return nullptr;
  }

  virtual std::shared_ptr<RegularizePtdwAgent>
  CreateRegularizePtdwAgent(const Batch& batch, const ProcessBatchesArgs& args, double tau) {
    return nullptr;
  }

  virtual bool RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                             const ::artm::core::PhiMatrix& n_wt,
                             ::artm::core::PhiMatrix* result) { return false; }

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize() {
    return google::protobuf::RepeatedPtrField<std::string>();
  }

  virtual google::protobuf::RepeatedPtrField<std::string> class_ids_to_regularize() {
    return google::protobuf::RepeatedPtrField<std::string>();
  }

  // Attempt to reconfigure an existing regularizer.
  // Returns true if succeeded, and false if the caller must recreate the regularizer from scratch
  // via constructor.
  virtual bool Reconfigure(const RegularizerConfig& config) { return false; }

  std::shared_ptr< ::artm::core::Dictionary> dictionary(const std::string& dictionary_name);
};

}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_INTERFACE_H_
