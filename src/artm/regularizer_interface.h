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
#include "artm/core/exceptions.h"
#include "artm/core/topic_model.h"

#include "glog/logging.h"

namespace artm {

namespace core {
  // Forward declarations
  class Regularizable;
  template<typename K, typename T> class ThreadSafeCollectionHolder;
  typedef std::map<artm::core::Token, ::artm::DictionaryEntry> DictionaryMap;
  typedef ThreadSafeCollectionHolder<std::string, DictionaryMap> ThreadSafeDictionaryCollection;
}

class RegularizerInterface {
 public:
  RegularizerInterface() : dictionaries_(nullptr) {}
  virtual ~RegularizerInterface() { }

  virtual bool RegularizeTheta(const Batch& batch,
                               const ModelConfig& model_config,
                               int inner_iter,
                               double tau,
                               ::artm::utility::DenseMatrix<float>* theta,
                               std::shared_ptr<const artm::core::TopicModel> topic_model) { return true; }

  virtual bool RegularizePhi(::artm::core::Regularizable* topic_model, double tau) { return true; }

  // Attempt to reconfigure an existing regularizer.
  // Returns true if succeeded, and false if the caller must recreate the regularizer from scratch
  // via constructor.
  virtual bool Reconfigure(const RegularizerConfig& config) { return false; }

  virtual void SerializeInternalState(RegularizerInternalState* regularizer_state) {
    BOOST_THROW_EXCEPTION(artm::core::InvalidOperation(
      "This regularizer has no internal state that can be retrieved."));
  }

  std::shared_ptr< ::artm::core::DictionaryMap> dictionary(const std::string& dictionary_name);
  void set_dictionaries(const ::artm::core::ThreadSafeDictionaryCollection* dictionaries);

 private:
  const ::artm::core::ThreadSafeDictionaryCollection* dictionaries_;
};

}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_INTERFACE_H_
