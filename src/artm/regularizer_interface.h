// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <vector>
#include <map>
#include <memory>
#include <string>

#include "artm/core/common.h"
#include "artm/core/dictionary.h"
#include "artm/core/exceptions.h"
#include "artm/utility/blas.h"

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
  virtual ~RegularizeThetaAgent() { }

  // Define how theta regularizer applies to an individual item.
  virtual void Apply(int item_index, int inner_iter, int topics_size, const float* n_td, float* r_td) const { }

  // The following method allows to calculate regularization for all elements of the theta matrix.
  // This seems convenient, however we do not recommended to overwrite this method.
  // The problem is that the default execution mode in BigARTM does not create full theta matrix even within a batch;
  // instead, each document withing a batch is processed one-by-one.
  // Therefore this option has effect only if opt_for_avx is set to false.
  // See code in processor.cc for more details.
  virtual void Apply(int inner_iter,
                     const ::artm::utility::LocalThetaMatrix<float>& n_td,
                     ::artm::utility::LocalThetaMatrix<float>* r_td) const;
};

class RegularizePtdwAgent {
 public:
  virtual ~RegularizePtdwAgent() { }
  virtual void Apply(int item_index, int inner_iter, ::artm::utility::LocalPhiMatrix<float>* ptdw) const = 0;
};

// RegularizerInterface is the base class for all regularizers in BigARTM.
// See any class in 'src/regularizer' folder for an example of how to implement new regularizer.
// Keep in mind that scres can be applied to either theta matrix, ptdw matrix or phi matrix.
// RegularizerInterface is, unfortunately, a base class for both of them.
// Hopefully we will refactor this at some point in the future.
// For performance reasons theta-regularizer involve 'RegularizerThetaAgent' or 'RegularizerPtdwAgent'.
// The idea is that you may analize the batch and store all information in your theta agent,
// so that later it can be applied efficiently to the items.
// A typical task to perform during construction of the agent is to analize the set of topics,
// to avoid looking at strings (topic_name) during processing of each individual item.
class RegularizerInterface {
 public:
  RegularizerInterface() { }
  virtual ~RegularizerInterface() { }

  virtual std::shared_ptr<RegularizeThetaAgent>
  CreateRegularizeThetaAgent(const Batch& batch, const ProcessBatchesArgs& args, float tau) {
    return nullptr;
  }

  virtual std::shared_ptr<RegularizePtdwAgent>
  CreateRegularizePtdwAgent(const Batch& batch, const ProcessBatchesArgs& args, float tau) {
    return nullptr;
  }

  // This is important to keep in mind when implementing new regularizers:
  //    n_wt and result are guarantied to have the same shape (e.i. topics and tokens)
  //    n_wt and p_wt are guarantied to share topics
  //    p_wt may have another set of tokens than n_wt (!).
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
