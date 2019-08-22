// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <map>
#include <vector>
#include <memory>
#include <string>

#include "artm/core/common.h"
#include "artm/core/phi_matrix.h"
#include "artm/core/instance.h"
#include "artm/core/token.h"

namespace artm {
namespace core {

typedef std::unordered_map<ClassId, std::vector<float>> Normalizers;

// PhiMatrixOperations contains helper methods to operate on PhiMatrix class.
class PhiMatrixOperations {
 public:
  // Extract protobuf message 'topic_model' from phi matrix
  static void RetrieveExternalTopicModel(
    const PhiMatrix& phi_matrix, const ::artm::GetTopicModelArgs& get_model_args,
    ::artm::TopicModel* topic_model);

  // Apply protobuf message 'topic_model' to phi_matrix
  static void ApplyTopicModelOperation(
    const ::artm::TopicModel& topic_model, float apply_weight, bool add_missing_tokens, PhiMatrix* phi_matrix);

  // Calculate phi matrix regularizers (r_wt)
  static void InvokePhiRegularizers(
    Instance* instance,
    const ::google::protobuf::RepeatedPtrField<RegularizerSettings>& regularizer_settings,
    const PhiMatrix& p_wt, const PhiMatrix& n_wt, PhiMatrix* r_wt);

  // For each ClassId finds a sum of all n_wt values for each topic with (optionally) regularizers r_wt
  static Normalizers FindNormalizers(const PhiMatrix& n_wt);
  static Normalizers FindNormalizers(const PhiMatrix& n_wt, const PhiMatrix& r_wt);

  // Produce normalized p_wt matrix from counters n_wt and (optionaly) regularizers r_wt
  static void FindPwt(const PhiMatrix& n_wt, PhiMatrix* p_wt);
  static void FindPwt(const PhiMatrix& n_wt, const PhiMatrix& r_wt, PhiMatrix* p_wt);

  // Checks whether two PhiMatrix instances has same set of tokens and topic names.
  // The order of the tokens and topics must also match.
  static bool HasEqualShape(const PhiMatrix& first, const PhiMatrix& second);
  static void AssignValue(float value, PhiMatrix* phi_matrix);

  // Convert ::artm::TopicModel to ::artm:Batch (pseudo-batch in hierarchical topic models)
  // Input object 'topic_model' could be modified by this operation.
  static void ConvertTopicModelToPseudoBatch(::artm::TopicModel* topic_model, ::artm::Batch* batch);
};

}  // namespace core
}  // namespace artm
