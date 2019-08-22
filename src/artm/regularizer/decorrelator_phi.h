/* Copyright 2017, Additive Regularization of Topic Models.

   Author: Murat Apishev (great-mel@yandex.ru)

   This class proceeds the decorrelation of topics in Phi matrix.
   The formula of M-step is
   
   p_wt \propto n_wt - tau * p_wt * \sum_{s \in T\t} p_ws.
   
   In case of using topic_pairs parameter the formula becomes a bit more complex:

   p_wt \propto n_wt - tau * p_wt * \sum_{s \in topic_pairs[t]} (p_ws * topic_pairs[t][s]).

   The parameters of the regularizer:
   - topic_names (the names of topics to regularize, empty == all)
   - class_ids (class ids to regularize, empty == all)
   - transaction_typenames (transaction typenames to regularize, empty == all)
   - topic_pairs (pair of topic names with value for their decorrelation,
                  empty == simple case usage)

Note, that in case of topic_pairs usage the topic_names parameter will be ignored.
*/

#pragma once

#include <unordered_map>
#include <string>

#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

typedef std::unordered_map<std::string, std::unordered_map<std::string, float> > TopicMap;

class DecorrelatorPhi : public RegularizerInterface {
 public:
  explicit DecorrelatorPhi(const DecorrelatorPhiConfig& config)
    : config_(config) {
    UpdateTopicPairs(config);
  }

  virtual bool RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                             const ::artm::core::PhiMatrix& n_wt,
                             ::artm::core::PhiMatrix* result);

  virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();
  virtual google::protobuf::RepeatedPtrField<std::string> class_ids_to_regularize();

  virtual bool Reconfigure(const RegularizerConfig& config);

 private:
  void UpdateTopicPairs(const DecorrelatorPhiConfig& config);

  DecorrelatorPhiConfig config_;
  TopicMap topic_pairs_;
};

}  // namespace regularizer
}  // namespace artm
