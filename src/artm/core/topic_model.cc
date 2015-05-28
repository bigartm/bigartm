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

TopicModel::~TopicModel() {
  Clear();
}

void TopicModel::Clear() {
  n_wt_.Clear();
  p_wt_.Clear();
}

void TopicModel::ApplyTopicModelOperation(const ::artm::TopicModel& topic_model, float apply_weight) {
  if (!Helpers::Validate(topic_model, /* throw_error=*/ false)) return;

  const bool use_sparse_format = (topic_model.topic_index_size() > 0);
  const int this_topic_size = this->topic_size();
  std::vector<int> target_topic_index;
  if (topic_model.topic_name_size() > 0) {
    bool ok = false;
    for (auto& topic_name : topic_model.topic_name()) {
      int index = repeated_field_index_of(this->topic_name(), topic_name);
      target_topic_index.push_back(index);
      if (index != -1) ok = true;
    }
    if (!ok) {
      LOG(ERROR) << "None of TopicModel.topic_name match topic names in target model";
      return;
    }
  } else {
    if (this->topic_size() != topic_model.topics_count())
      BOOST_THROW_EXCEPTION(InvalidOperation("Mismatch between target topics_count and TopicModel.topics_count"));
    for (int i = 0; i < topic_model.topics_count(); ++i)
      target_topic_index.push_back(i);
  }

  bool optimized_execution = false;
  if ((apply_weight == 1.0f) && (target_topic_index.size() == this_topic_size)) {
    bool ok = true;
    for (int topic_index = 0; topic_index < target_topic_index.size(); ++topic_index) {
      if (target_topic_index[topic_index] != topic_index)
        ok = false;
    }
    optimized_execution = ok;
  }

  for (int token_index = 0; token_index < topic_model.token_size(); ++token_index) {
    const std::string& token_keyword = topic_model.token(token_index);
    const ClassId& class_id = topic_model.class_id(token_index);
    Token token(class_id, token_keyword);
    const FloatArray& counters = topic_model.token_weights(token_index);
    const IntArray* sparse_topic_index = use_sparse_format ? &topic_model.topic_index(token_index) : nullptr;
    const bool use_sparse_format_local = (sparse_topic_index != nullptr) && (sparse_topic_index->value_size() > 0);

    TopicModel_OperationType operation_type = topic_model.operation_type(token_index);
    int current_token_id = token_id(token);

    float* target;
    switch (operation_type) {
      case TopicModel_OperationType_Initialize:
        // Add new tokens discovered by processor
        if (current_token_id == -1)
          this->AddToken(token, true);
        break;

      case TopicModel_OperationType_Increment:
        if (current_token_id == -1)
          current_token_id = this->AddToken(token, false);
        target = n_wt_[current_token_id];

        if (optimized_execution && !use_sparse_format_local && (counters.value_size() == this_topic_size)) {
          for (int topic_index = 0; topic_index < this_topic_size; ++topic_index)
            target[topic_index] += counters.value(topic_index);
          break;
        }

        for (int i = 0; i < counters.value_size(); ++i) {
          int topic_index = use_sparse_format_local ? sparse_topic_index->value(i) : i;
          assert(topic_index < target_topic_index.size());
          if (target_topic_index[topic_index] == -1)
            continue;
          target[target_topic_index[topic_index]] += apply_weight * counters.value(i);
        }
        break;

      case TopicModel_OperationType_Overwrite:
        if (current_token_id == -1)
          current_token_id = this->AddToken(token, false);
        target = n_wt_[current_token_id];
        for (int i = 0; i < counters.value_size(); ++i) {
          int topic_index = use_sparse_format_local ? sparse_topic_index->value(i) : i;
          assert(topic_index < target_topic_index.size());
          if (target_topic_index[topic_index] == -1)
            continue;
          target[target_topic_index[topic_index]] = counters.value(i);
        }
        break;

      case TopicModel_OperationType_Remove:
        this->RemoveToken(token);
        break;

      case TopicModel_OperationType_Ignore:
        // ignore token == do nothing
        break;

      default:
        BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
          "ModelIncrement.operation_type", operation_type));
    }
  }
}

void TopicModel::RetrieveExternalTopicModel(const ::artm::GetTopicModelArgs& get_model_args,
                                            ::artm::TopicModel* topic_model) const {
  if (token_size() == 0)
    return;

  const bool use_pwt = (get_model_args.request_type() == GetTopicModelArgs_RequestType_Pwt);
  const bool use_nwt = (get_model_args.request_type() == GetTopicModelArgs_RequestType_Nwt);
  if (!use_pwt && !use_nwt)
    BOOST_THROW_EXCEPTION(artm::core::InvalidOperation("Invalid GetTopicModelArgs_RequestType"));
  if (use_pwt && p_wt_.empty())
    BOOST_THROW_EXCEPTION(artm::core::InvalidOperation("pwt is not available in this TopicModel"));
  if (use_nwt && n_wt_.empty())
    BOOST_THROW_EXCEPTION(artm::core::InvalidOperation("nwt is not available in this TopicModel"));

  PhiMatrixOperations::RetrieveExternalTopicModel(use_pwt ? p_wt_ : n_wt_, get_model_args, topic_model);
}

int TopicModel::AddToken(const Token& token, bool random_init) {
  return n_wt_.AddToken(token, random_init);
}

void TopicModel::RemoveToken(const Token& token) {
  n_wt_.RemoveToken(token);
}

int TopicModel::topic_size() const {
  return n_wt_.topic_size();
}

google::protobuf::RepeatedPtrField<std::string> TopicModel::topic_name() const {
  return n_wt_.topic_name();
}

ModelName TopicModel::model_name() const {
  return n_wt_.model_name();
}

void TopicModel::CalcPwt() {
  p_wt_.Reshape(n_wt_);
  PhiMatrixOperations::FindPwt(n_wt_, &p_wt_);
}

void TopicModel::CalcPwt(const PhiMatrix& r_wt) {
  p_wt_.Reshape(n_wt_);
  PhiMatrixOperations::FindPwt(n_wt_, &p_wt_);
}

}  // namespace core
}  // namespace artm
