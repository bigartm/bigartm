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

#include "glog/logging.h"

namespace artm {

namespace core {
  // Forward declarations
  class Regularizable;
  template<typename K, typename T> class ThreadSafeCollectionHolder;
  typedef std::map<artm::core::Token, ::artm::DictionaryEntry> DictionaryMap;
  typedef ThreadSafeCollectionHolder<std::string, DictionaryMap> ThreadSafeDictionaryCollection;
}

class RegularizeThetaAgent {
 public:
  virtual ~RegularizeThetaAgent() {}
  virtual void Apply(int item_index, int inner_iter, int topics_size, float* theta) { return; }
};

class RegularizeThetaAgentCollection : public RegularizeThetaAgent {
 private:
  std::vector<std::shared_ptr<RegularizeThetaAgent>> agents_;

 public:
  void AddAgent(std::shared_ptr<RegularizeThetaAgent> agent) { agents_.push_back(agent); }
  virtual void Apply(int item_index, int inner_iter, int topics_size, float* theta);
};

class NormalizeThetaAgent : public RegularizeThetaAgent {
 public:
  virtual void Apply(int item_index, int inner_iter, int topics_size, float* theta);
};

class RegularizerInterface {
 public:
  RegularizerInterface() : dictionaries_(nullptr) {}
  virtual ~RegularizerInterface() { }

  virtual std::shared_ptr<RegularizeThetaAgent>
  CreateRegularizeThetaAgent(const Batch& batch, const ModelConfig& model_config, double tau) {
    return nullptr;
  }

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
