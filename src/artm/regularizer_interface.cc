// Copyright 2017, Additive Regularization of Topic Models.

#include "artm/regularizer_interface.h"

#include "artm/core/dictionary.h"

namespace artm {

std::shared_ptr< ::artm::core::Dictionary> RegularizerInterface::dictionary(const std::string& dictionary_name) {
  return ::artm::core::ThreadSafeDictionaryCollection::singleton().get(dictionary_name);
}

void RegularizeThetaAgent::Apply(int inner_iter,
                                 const ::artm::utility::LocalThetaMatrix<float>& n_td,
                                 ::artm::utility::LocalThetaMatrix<float>* r_td) const {
  if (!n_td.is_equal_size(*r_td)) {
    LOG(ERROR) << "Size mismatch between n_td and r_rd";
  }

  // The default implementation just calls Apply() for all items
  // Custom implementation may implement any other method that jointly acts on all items within a batch.
  for (int item_index = 0; item_index < n_td.num_items(); ++item_index) {
    this->Apply(item_index, inner_iter, n_td.num_topics(), &(n_td)(0, item_index), &(*r_td)(0, item_index));  // NOLINT
  }
}

}  // namespace artm
