// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_TOPIC_MODEL_H_
#define SRC_ARTM_CORE_TOPIC_MODEL_H_

#include <assert.h>

#include <algorithm>
#include <map>
#include <unordered_map>
#include <vector>
#include <set>
#include <string>

#include "boost/utility.hpp"
#include "boost/uuid/uuid.hpp"

#include "artm/messages.pb.h"

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"
#include "artm/core/phi_matrix.h"
#include "artm/core/dense_phi_matrix.h"
#include "artm/utility/blas.h"

namespace artm {
namespace core {

class TopicModel;

// A class representing a topic model.
// - ::artm::core::TopicModel is an internal representation, used in Processor and Merger.
//   It supports efficient lookup of words in the matrix.
// - ::artm::TopicModel is an external representation, implemented as protobuf message.
//   It is used to transfer model back to the user.
class TopicModel {
 public:
  explicit TopicModel(const ModelName& model_name,
                      const google::protobuf::RepeatedPtrField<std::string>& topic_name);

  void Clear();
  virtual ~TopicModel();

  void RetrieveExternalTopicModel(
    const ::artm::GetTopicModelArgs& get_model_args,
    ::artm::TopicModel* topic_model) const;

  void ApplyTopicModelOperation(const ::artm::TopicModel& topic_model, float apply_weight);

  void RemoveToken(const Token& token);
  int  AddToken(const Token& token, bool random_init = true);

  void CalcPwt();
  void CalcPwt(const PhiMatrix& r_wt);
  const PhiMatrix& GetPwt() const { return p_wt_; }

  ModelName model_name() const;

  int token_size() const { return n_wt_.token_size(); }
  int topic_size() const;
  google::protobuf::RepeatedPtrField<std::string> topic_name() const;

  bool has_token(const Token& token) const { return n_wt_.has_token(token); }
  int token_id(const Token& token) const { return n_wt_.token_index(token); }
  const Token& token(int index) const { return n_wt_.token(index); }

  const PhiMatrix& Nwt () const { return n_wt_; }

 private:
  DensePhiMatrix n_wt_;  // vector of length tokens_count
  DensePhiMatrix p_wt_;  // normalized matrix
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_TOPIC_MODEL_H_
