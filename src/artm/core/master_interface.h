// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_MASTER_INTERFACE_H_
#define SRC_ARTM_CORE_MASTER_INTERFACE_H_

#include <string>

#include "artm/messages.pb.h"
#include "artm/core/common.h"
#include "artm/core/template_manager.h"

namespace artm {
namespace core {

class MasterInterface {
 public:
  virtual ~MasterInterface() {}

  virtual int id() const = 0;

  virtual void Reconfigure(const MasterComponentConfig& config) = 0;

  virtual void CreateOrReconfigureModel(const ModelConfig& config) = 0;
  virtual void DisposeModel(ModelName model_name) = 0;

  virtual void CreateOrReconfigureRegularizer(const RegularizerConfig& config) = 0;
  virtual void DisposeRegularizer(const std::string& name) = 0;

  virtual void CreateOrReconfigureDictionary(const DictionaryConfig& config) = 0;
  virtual void DisposeDictionary(const std::string& name) = 0;

  virtual void OverwriteTopicModel(const ::artm::TopicModel& topic_model) = 0;
  virtual bool RequestTopicModel(const ::artm::GetTopicModelArgs& get_model_args,
                                 ::artm::TopicModel* topic_model) = 0;
  virtual void RequestRegularizerState(RegularizerName regularizer_name,
                                       ::artm::RegularizerInternalState* regularizer_state) = 0;
  virtual bool RequestThetaMatrix(const ::artm::GetThetaMatrixArgs& get_theta_args,
                                  ::artm::ThetaMatrix* theta_matrix) = 0;
  virtual bool RequestScore(const ::artm::GetScoreValueArgs& get_score_args,
                            ScoreData* score_data) = 0;

  virtual void AddBatch(const ::artm::AddBatchArgs& args) = 0;
  virtual void InvokeIteration(const ::artm::InvokeIterationArgs& args) = 0;
  virtual bool WaitIdle(const ::artm::WaitIdleArgs& args) = 0;
  virtual void SynchronizeModel(const SynchronizeModelArgs& args) = 0;
  virtual void InitializeModel(const InitializeModelArgs& args) = 0;
};

typedef TemplateManager<MasterInterface> MasterComponentManager;

}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_MASTER_INTERFACE_H_
