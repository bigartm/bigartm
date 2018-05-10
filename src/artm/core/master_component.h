// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"

#include "artm/core/common.h"
#include "artm/core/dictionary.h"
#include "artm/core/thread_safe_holder.h"

namespace artm {

class RegularizerInterface;

namespace core {

class ArtmExecutor;
class Instance;
class TopicModel;
class BatchManager;
class ScoreManager;

// MasterComponent is an orchestration layer that implements most functionality in BigARTM core.
// Most methods in c_interface (the API) call some methods in MasterComponent.
// This clas smight be renamed to MasterModel, because it also implements methods
// ArtmCreateMasterModel, ArtmFitOfflineMasterModel, ArtmFitOnlineMasterModel, ArtmTransformMasterModel.
// Each instance of MasterComponent defines its own namespace for phi matrices, regularizers, scores.
// All data fiels of MasterComponent are stored in the 'Instance' class.
// If you are familiar with Pimpl idiom you may think of 'Instance' as of 'MasterComponentPimpl'.
class MasterComponent : boost::noncopyable {
 public:
  ~MasterComponent();

  std::shared_ptr<MasterModelConfig> config() const;

  explicit MasterComponent(const MasterModelConfig& config);
  std::shared_ptr<MasterComponent> Duplicate() const;

  // REQUEST functionality
  void Request(::artm::MasterModelConfig* result);
  void Request(const GetTopicModelArgs& args, ::artm::TopicModel* result);
  void Request(const GetTopicModelArgs& args, ::artm::TopicModel* result, std::string* external);
  void Request(const GetThetaMatrixArgs& args, ThetaMatrix* result);
  void Request(const GetThetaMatrixArgs& args, ThetaMatrix* result, std::string* external);
  void Request(const TransformMasterModelArgs& args, ThetaMatrix* result);
  void Request(const TransformMasterModelArgs& args, ThetaMatrix* result, std::string* external);
  void Request(const GetScoreValueArgs& args, ScoreData* result);
  void Request(const GetScoreArrayArgs& args, ScoreArray* result);
  void Request(const ProcessBatchesArgs& args, ProcessBatchesResult* result);
  void Request(const ProcessBatchesArgs& args, ProcessBatchesResult* result, std::string* external);
  void Request(const GetDictionaryArgs& args, DictionaryData* result);
  void Request(const GetMasterComponentInfoArgs& args, MasterComponentInfo* result);

  // EXECUTE functionality
  void MergeModel(const MergeModelArgs& args);
  void RegularizeModel(const RegularizeModelArgs& args);
  void NormalizeModel(const NormalizeModelArgs& args);
  void ImportDictionary(const ImportDictionaryArgs& args);
  void ExportDictionary(const ExportDictionaryArgs& args);
  void ImportBatches(const ImportBatchesArgs& args);
  void ExportModel(const ExportModelArgs& args);
  void ImportModel(const ImportModelArgs& args);
  void InitializeModel(const InitializeModelArgs& args);
  void FitOnline(const FitOnlineMasterModelArgs& args);
  void FitOffline(const FitOfflineMasterModelArgs& args);
  void FilterDictionary(const FilterDictionaryArgs& args);
  void GatherDictionary(const GatherDictionaryArgs& args);
  void ClearThetaCache(const ClearThetaCacheArgs& args);
  void ClearScoreCache(const ClearScoreCacheArgs& args);
  void ClearScoreArrayCache(const ClearScoreArrayCacheArgs& args);
  void ExportScoreTracker(const ExportScoreTrackerArgs& args);
  void ImportScoreTracker(const ImportScoreTrackerArgs& args);

  // DISPOSE functionality
  void DisposeModel(const std::string& name);
  void DisposeRegularizer(const std::string& name);
  void DisposeDictionary(const std::string& name);
  void DisposeBatch(const std::string& name);

  // Other ad-hoc functionality
  void AsyncRequestProcessBatches(const ProcessBatchesArgs& process_batches_args,
                                  BatchManager *batch_manager);

  // Reconfigures topic model if already exists, otherwise creates a new model.
  void OverwriteTopicModel(const ::artm::TopicModel& topic_model);

  void ReconfigureMasterModel(const MasterModelConfig& config);
  void ReconfigureTopicName(const MasterModelConfig& config);

  void CreateOrReconfigureRegularizer(const RegularizerConfig& config);

  void CreateDictionary(const DictionaryData& data);

  void AttachModel(const AttachModelArgs& args, int address_length, float* address);

 private:
  friend class ArtmExecutor;

  MasterComponent(const MasterComponent& rhs);
  MasterComponent& operator=(const MasterComponent&);

  void RequestProcessBatchesImpl(const ProcessBatchesArgs& process_batches_args,
                                 BatchManager* batch_manager, bool async,
                                 ScoreManager* score_manager,
                                 ::artm::ThetaMatrix* theta_matrix);

  void CreateOrReconfigureMasterComponent(const MasterModelConfig& config, bool reconfigure,
                                          bool change_topic_name);

  void AddDictionary(std::shared_ptr<Dictionary> dictionary);

  std::shared_ptr<Instance> instance_;
};

}  // namespace core
}  // namespace artm
