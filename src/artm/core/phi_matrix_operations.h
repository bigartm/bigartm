// Copyright 2015, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_PHI_MATRIX_OPERATIONS_H_
#define SRC_ARTM_CORE_PHI_MATRIX_OPERATIONS_H_

#include <map>
#include <vector>

#include "artm/core/common.h"
#include "artm/core/phi_matrix.h"
#include "artm/messages.pb.h"

namespace artm {
namespace core {

class PhiMatrixOperations {
 public:
  static void RetrieveExternalTopicModel(
    const PhiMatrix& phi_matrix, const ::artm::GetTopicModelArgs& get_model_args,
    ::artm::TopicModel* topic_model);

  static void ApplyTopicModelOperation(
    const ::artm::TopicModel& topic_model, float apply_weight, PhiMatrix* phi_matrix);

  static std::map<ClassId, std::vector<float> > FindNormalizers(const PhiMatrix& n_wt);
  static std::map<ClassId, std::vector<float> > FindNormalizers(const PhiMatrix& n_wt, const PhiMatrix& r_wt);

  static void FindPwt(const PhiMatrix& n_wt, PhiMatrix* p_wt);
  static void FindPwt(const PhiMatrix& n_wt, const PhiMatrix& r_wt, PhiMatrix* p_wt);
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_PHI_MATRIX_OPERATIONS_H_
