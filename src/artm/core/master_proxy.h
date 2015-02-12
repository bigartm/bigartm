// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_MASTER_PROXY_H_
#define SRC_ARTM_CORE_MASTER_PROXY_H_

#include <string>

#include "boost/thread.hpp"
#include "rpcz/application.hpp"

#include "artm/messages.pb.h"
#include "artm/core/internals.pb.h"
#include "artm/core/internals.rpcz.h"
#include "artm/core/master_interface.h"
#include "artm/core/template_manager.h"

namespace artm {
namespace core {

class MasterProxy : boost::noncopyable, public MasterInterface {
 public:
  ~MasterProxy();

  virtual int id() const;

  virtual void Reconfigure(const MasterComponentConfig& config);

  virtual void CreateOrReconfigureModel(const ModelConfig& config);
  virtual void DisposeModel(ModelName model_name);

  virtual void CreateOrReconfigureRegularizer(const RegularizerConfig& config);
  virtual void DisposeRegularizer(const std::string& name);

  virtual void CreateOrReconfigureDictionary(const DictionaryConfig& config);
  virtual void DisposeDictionary(const std::string& name);

  virtual void OverwriteTopicModel(const ::artm::TopicModel& topic_model);
  virtual bool RequestTopicModel(const ::artm::GetTopicModelArgs& get_model_args,
                                 ::artm::TopicModel* topic_model);
  virtual void RequestRegularizerState(RegularizerName regularizer_name,
                                       ::artm::RegularizerInternalState* regularizer_state);
  virtual bool RequestThetaMatrix(const GetThetaMatrixArgs& get_theta_args,
                                  ::artm::ThetaMatrix* theta_matrix);
  virtual bool RequestScore(const GetScoreValueArgs& get_score_args,
                            ScoreData* score_data);

  virtual bool AddBatch(const AddBatchArgs& args);
  virtual void InvokeIteration(const InvokeIterationArgs& args);
  virtual bool WaitIdle(const WaitIdleArgs& args);
  virtual void SynchronizeModel(const SynchronizeModelArgs& args);
  virtual void InitializeModel(const InitializeModelArgs& args);

 private:
  friend class TemplateManager<MasterInterface>;

  // All master components must be created via TemplateManager.
  MasterProxy(int id, const MasterProxyConfig& config);

  int id_;

  int communication_timeout_;
  int polling_frequency_;
  std::unique_ptr<rpcz::application> application_;
  std::shared_ptr<artm::core::NodeControllerService_Stub> node_controller_service_proxy_;
};

}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_MASTER_PROXY_H_
