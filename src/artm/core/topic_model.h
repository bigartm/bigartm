// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_TOPIC_MODEL_H_
#define SRC_ARTM_CORE_TOPIC_MODEL_H_

#include <assert.h>

#include <string>

#include "boost/utility.hpp"

#include "artm/messages.pb.h"

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"
#include "artm/core/phi_matrix.h"
#include "artm/core/dense_phi_matrix.h"

namespace artm {
namespace core {

// A strange wrapper for (nwt, pwt) pair.
// nwt is always present; pwt can be absent.
class TopicModel {
 public:
  explicit TopicModel(const ModelName& model_name,
                      const google::protobuf::RepeatedPtrField<std::string>& topic_name);

  void RetrieveExternalTopicModel(
    const ::artm::GetTopicModelArgs& get_model_args,
    ::artm::TopicModel* topic_model) const;

  void CalcPwt();
  void CalcPwt(const PhiMatrix& r_wt);
  const PhiMatrix& GetPwt() const { return p_wt_; }
  const PhiMatrix& GetNwt () const { return n_wt_; }

  PhiMatrix* mutable_pwt() { return &p_wt_; }
  PhiMatrix* mutable_nwt() { return &n_wt_; }

  ModelName model_name() const { return n_wt_.model_name(); }
  int token_size() const { return n_wt_.token_size(); }
  int topic_size() const { return n_wt_.topic_size(); }
  google::protobuf::RepeatedPtrField<std::string> topic_name() const { return n_wt_.topic_name(); }

  bool has_token(const Token& token) const { return n_wt_.has_token(token); }
  int token_id(const Token& token) const { return n_wt_.token_index(token); }
  const Token& token(int index) const { return n_wt_.token(index); }

 private:
  DensePhiMatrix n_wt_;
  DensePhiMatrix p_wt_;  // normalized matrix
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_TOPIC_MODEL_H_
