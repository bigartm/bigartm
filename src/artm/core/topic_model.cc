// Copyright 2014, Additive Regularization of Topic Models.

#include <artm/core/topic_model.h>

#include <assert.h>
#include <math.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/lexical_cast.hpp"
#include "boost/uuid/string_generator.hpp"
#include "boost/uuid/uuid_io.hpp"

#include "glog/logging.h"

#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/protobuf_helpers.h"
#include "artm/core/phi_matrix_operations.h"

namespace artm {
namespace core {

TopicModel::TopicModel(const ModelName& model_name,
                       const google::protobuf::RepeatedPtrField<std::string>& topic_name)
    : n_wt_(model_name, topic_name),
      p_wt_(model_name, topic_name) {
}

void TopicModel::RetrieveExternalTopicModel(const ::artm::GetTopicModelArgs& get_model_args,
                                            ::artm::TopicModel* topic_model) const {
  const bool use_pwt = (get_model_args.request_type() == GetTopicModelArgs_RequestType_Pwt);
  const bool use_nwt = (get_model_args.request_type() == GetTopicModelArgs_RequestType_Nwt);
  if (!use_pwt && !use_nwt)
    BOOST_THROW_EXCEPTION(artm::core::InvalidOperation("Invalid GetTopicModelArgs_RequestType"));
  if (use_pwt && (p_wt_.token_size() == 0))
    BOOST_THROW_EXCEPTION(artm::core::InvalidOperation("pwt is not calculated for this TopicModel"));

  PhiMatrixOperations::RetrieveExternalTopicModel(use_pwt ? p_wt_ : n_wt_, get_model_args, topic_model);
}

void TopicModel::CalcPwt() {
  p_wt_.Reshape(n_wt_);
  PhiMatrixOperations::FindPwt(n_wt_, &p_wt_);
}

void TopicModel::CalcPwt(const PhiMatrix& r_wt) {
  p_wt_.Reshape(n_wt_);
  PhiMatrixOperations::FindPwt(n_wt_, r_wt, &p_wt_);
}

}  // namespace core
}  // namespace artm
