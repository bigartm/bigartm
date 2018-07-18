/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)

   This class proceeds sparsing Phi matrix by set value.
   It is not a usual regularizer, it's a tool to sparse as
   many elements in Phi, as you need. You can sparse by columns
   or by rows.
   
   The parameters of the regularizer:
   - topic_names (the names of topics to regularize, empty == all)
   - class_id (class id to regularize, required)
   - transaction_typename (transaction typename to regularize,
                           if empty -> == DefaultTransactionTypeName)
   - mode (by rows or by columns, default == by colmns (e.g. toppics))
   - max_elements_count (the number of most probable elements to be saved
     in each row/column, other should be set to zero)
   - probability_threshold (if the sum of values of n elements,
     n < max_elements_count, have already reached this value, than stop
     and zero all others in this row/column)
*/

#pragma once

#include <string>

#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class SpecifiedSparsePhi : public RegularizerInterface {
 public:
  explicit SpecifiedSparsePhi(const SpecifiedSparsePhiConfig& config) : config_(config) { }

  virtual bool RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                             const ::artm::core::PhiMatrix& n_wt,
                             ::artm::core::PhiMatrix* result);

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();
  virtual google::protobuf::RepeatedPtrField<std::string> class_ids_to_regularize();

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  SpecifiedSparsePhiConfig config_;
};

}  // namespace regularizer
}  // namespace artm
