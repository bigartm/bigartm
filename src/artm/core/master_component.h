// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_MASTER_COMPONENT_H_
#define SRC_ARTM_CORE_MASTER_COMPONENT_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"

#include "artm/messages.pb.h"

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"
#include "artm/core/template_manager.h"
#include "artm/core/thread_safe_holder.h"

namespace artm {

class RegularizerInterface;

namespace core {

class Instance;
class TopicModel;
class Score;

class MasterComponent : boost::noncopyable {
 public:
  ~MasterComponent();

  int id() const;
  std::shared_ptr<MasterComponentConfig> config() const;

  // Retrieves topic model.
  // Returns true if succeeded, and false if model_name hasn't been found.
  bool RequestTopicModel(const ::artm::GetTopicModelArgs& get_model_args,
                         ::artm::TopicModel* topic_model);
  void RequestRegularizerState(RegularizerName regularizer_name,
                               ::artm::RegularizerInternalState* regularizer_state);
  bool RequestThetaMatrix(const GetThetaMatrixArgs& get_theta_args,
                          ::artm::ThetaMatrix* theta_matrix);
  bool RequestScore(const GetScoreValueArgs& get_score_args,
                    ScoreData* score_data);

  void RequestProcessBatches(const ProcessBatchesArgs& process_batches_args,
                             ProcessBatchesResult* process_batches_result);
  void MergeModel(const MergeModelArgs& merge_model_args);
  void RegularizeModel(const RegularizeModelArgs& regularize_model_args);
  void NormalizeModel(const NormalizeModelArgs& normalize_model_args);

  // Reconfigures topic model if already exists, otherwise creates a new model.
  void CreateOrReconfigureModel(const ModelConfig& config);
  void OverwriteTopicModel(const ::artm::TopicModel& topic_model);

  void DisposeModel(ModelName model_name);
  void Reconfigure(const MasterComponentConfig& config);

  void CreateOrReconfigureRegularizer(const RegularizerConfig& config);
  void DisposeRegularizer(const std::string& name);

  void CreateOrReconfigureDictionary(const DictionaryConfig& config);
  void DisposeDictionary(const std::string& name);
  void ImportDictionary(const ImportDictionaryArgs& args);

  // Returns false if BigARTM is still processing the collection, otherwise true.
  bool WaitIdle(const WaitIdleArgs& args);
  void InvokeIteration(const InvokeIterationArgs& args);
  void SynchronizeModel(const SynchronizeModelArgs& args);
  void ExportModel(const ExportModelArgs& args);
  void ImportModel(const ImportModelArgs& args);
  void AttachModel(const AttachModelArgs& args, int address_length, float* address);
  void InitializeModel(const InitializeModelArgs& args);
  bool AddBatch(const AddBatchArgs& args);

 private:
  friend class TemplateManager<MasterComponent>;

  // All master components must be created via TemplateManager.
  MasterComponent(int id, const MasterComponentConfig& config);
  MasterComponent(int id, const MasterComponent& rhs);
  MasterComponent& operator=(const MasterComponent&);

  int master_id_;

  std::shared_ptr<Instance> instance_;
};

typedef TemplateManager<MasterComponent> MasterComponentManager;

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_MASTER_COMPONENT_H_
