// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/regularizer_interface.h"

#include "artm/core/dictionary.h"

namespace artm {

std::shared_ptr< ::artm::core::Dictionary> RegularizerInterface::dictionary(const std::string& dictionary_name) {
  return ::artm::core::ThreadSafeDictionaryCollection::singleton().get(dictionary_name);
}

}  // namespace artm
