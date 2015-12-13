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

  // REQUEST functionality
  void Request(const GetTopicModelArgs& args, ::artm::TopicModel* result);
  void Request(const GetTopicModelArgs& args, ::artm::TopicModel* result, std::string* external);
  void Request(const GetThetaMatrixArgs& args, ThetaMatrix* result);
  void Request(const GetThetaMatrixArgs& args, ThetaMatrix* result, std::string* external);
  void Request(const GetScoreValueArgs& args, ScoreData* result);
  void Request(const ProcessBatchesArgs& args, ProcessBatchesResult* result);
  void Request(const ProcessBatchesArgs& args, ProcessBatchesResult* result, std::string* external);
  void Request(const GetDictionaryArgs& args, DictionaryData* result);
  void Request(const GetMasterComponentInfoArgs& args, MasterComponentInfo* result);
  void Request(const GetRegularizerStateArgs& args, RegularizerInternalState* regularizer_state);

  // EXECUTE functionality
  void MergeModel(const MergeModelArgs& args);
  void RegularizeModel(const RegularizeModelArgs& args);
  void NormalizeModel(const NormalizeModelArgs& args);
  void ImportDictionary(const ImportDictionaryArgs& args);
  void ExportDictionary(const ExportDictionaryArgs& args);
  void ImportBatches(const ImportBatchesArgs& args);
  void InvokeIteration(const InvokeIterationArgs& args);
  void SynchronizeModel(const SynchronizeModelArgs& args);
  void ExportModel(const ExportModelArgs& args);
  void ImportModel(const ImportModelArgs& args);
  void InitializeModel(const InitializeModelArgs& args);
  void FilterDictionary(const FilterDictionaryArgs& args);
  void GatherDictionary(const GatherDictionaryArgs& args);
  bool AddBatch(const AddBatchArgs& args);

  // DISPOSE functionality
  void DisposeModel(const std::string& name);
  void DisposeRegularizer(const std::string& name);
  void DisposeDictionary(const std::string& name);
  void DisposeBatch(const std::string& name);

  // Other ad-hoc functionality
  void AsyncRequestProcessBatches(const ProcessBatchesArgs& process_batches_args,
                                  BatchManager *batch_manager);

  // Reconfigures topic model if already exists, otherwise creates a new model.
  void CreateOrReconfigureModel(const ModelConfig& config);
  void OverwriteTopicModel(const ::artm::TopicModel& topic_model);

  void Reconfigure(const MasterComponentConfig& config);

  void CreateOrReconfigureRegularizer(const RegularizerConfig& config);

  void CreateDictionary(const DictionaryData& data);
  void AppendDictionary(const DictionaryData& data);

  // Returns false if BigARTM is still processing the collection, otherwise true.
  bool WaitIdle(const WaitIdleArgs& args);

  void AttachModel(const AttachModelArgs& args, int address_length, float* address);

 private:
  MasterComponent(const MasterComponent& rhs);
  MasterComponent& operator=(const MasterComponent&);

  void RequestProcessBatchesImpl(const ProcessBatchesArgs& process_batches_args,
                                 BatchManager* batch_manager, bool async,
                                 ProcessBatchesResult* process_batches_result);

  std::shared_ptr<Instance> instance_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_MASTER_COMPONENT_H_
