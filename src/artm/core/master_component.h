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
#include "artm/core/thread_safe_holder.h"

namespace artm {

class RegularizerInterface;

namespace core {

class Instance;
class TopicModel;
class Score;
class BatchManager;

class MasterComponent : boost::noncopyable {
 public:
  ~MasterComponent();

  std::shared_ptr<MasterComponentConfig> config() const;

  explicit MasterComponent(const MasterComponentConfig& config);
  std::shared_ptr<MasterComponent> Duplicate() const;

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
  void RequestMasterComponentInfo(MasterComponentInfo* master_info) const;

  void RequestProcessBatches(const ProcessBatchesArgs& process_batches_args,
                             ProcessBatchesResult* process_batches_result);

  void AsyncRequestProcessBatches(const ProcessBatchesArgs& process_batches_args,
                                  BatchManager *batch_manager);

  void RequestProcessBatchesImpl(const ProcessBatchesArgs& process_batches_args,
                                 BatchManager* batch_manager, bool async,
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

  void CreateDictionary(const DictionaryData& data);
  void AppendDictionary(const DictionaryData& data);
  void DisposeDictionary(const std::string& name);
  void ImportDictionary(const ImportDictionaryArgs& args);
  void ExportDictionary(const ExportDictionaryArgs& args);
  void RequestDictionary(const GetDictionaryArgs& args, DictionaryData* result);

  void ImportBatches(const ImportBatchesArgs& args);
  void DisposeBatches(const DisposeBatchesArgs& args);

  // Returns false if BigARTM is still processing the collection, otherwise true.
  bool WaitIdle(const WaitIdleArgs& args);
  void InvokeIteration(const InvokeIterationArgs& args);
  void SynchronizeModel(const SynchronizeModelArgs& args);
  void ExportModel(const ExportModelArgs& args);
  void ImportModel(const ImportModelArgs& args);
  void AttachModel(const AttachModelArgs& args, int address_length, float* address);
  void InitializeModel(const InitializeModelArgs& args);
  void FilterDictionary(const FilterDictionaryArgs& args);
  void GatherDictionary(const GatherDictionaryArgs& args);
  bool AddBatch(const AddBatchArgs& args);

 private:
  MasterComponent(const MasterComponent& rhs);
  MasterComponent& operator=(const MasterComponent&);

  std::shared_ptr<Instance> instance_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_MASTER_COMPONENT_H_
