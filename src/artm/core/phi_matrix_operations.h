// Copyright 2015, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_PHI_MATRIX_OPERATIONS_H_
#define SRC_ARTM_CORE_PHI_MATRIX_OPERATIONS_H_

#include <map>
#include <vector>
#include <memory>

#include "artm/core/common.h"
#include "artm/core/phi_matrix.h"
#include "artm/messages.pb.h"
#include "artm/core/instance_schema.h"

namespace artm {
namespace core {

class PhiMatrixOperations {
 public:
  static void RetrieveExternalTopicModel(
    const PhiMatrix& phi_matrix, const ::artm::GetTopicModelArgs& get_model_args,
    ::artm::TopicModel* topic_model);

  static void ApplyTopicModelOperation(
    const ::artm::TopicModel& topic_model, float apply_weight, PhiMatrix* phi_matrix);

  static void InvokePhiRegularizers(
    std::shared_ptr<InstanceSchema> schema,
    const ::google::protobuf::RepeatedPtrField<RegularizerSettings>& regularizer_settings,
    const PhiMatrix& p_wt, const PhiMatrix& n_wt, PhiMatrix* r_wt);

  static std::map<ClassId, std::vector<float> > FindNormalizers(const PhiMatrix& n_wt);
  static std::map<ClassId, std::vector<float> > FindNormalizers(const PhiMatrix& n_wt, const PhiMatrix& r_wt);

  static void FindPwt(const PhiMatrix& n_wt, PhiMatrix* p_wt);
  static void FindPwt(const PhiMatrix& n_wt, const PhiMatrix& r_wt, PhiMatrix* p_wt);

  static bool HasEqualShape(const PhiMatrix& first, const PhiMatrix& second);
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_PHI_MATRIX_OPERATIONS_H_
