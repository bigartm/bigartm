// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <string>
#include <vector>

#include "artm/core/common.h"
#include "artm/c_interface.h"
#include "artm/cpp_interface.h"

namespace artm {
namespace test {

// Defines additional APIs, not exposed through ::artm::MasterModel interface.
class Api {
 public:
  explicit Api(MasterModel& master_model) : master_model_(master_model) { }

  // Methods wrapping c_interface
  TopicModel AttachTopicModel(const AttachModelArgs& args, Matrix* matrix);
  ThetaMatrix ProcessBatches(const ProcessBatchesArgs& args);
  int AsyncProcessBatches(const ProcessBatchesArgs& args);
  int AwaitOperation(int operation_id);
  void MergeModel(const MergeModelArgs& args);
  void NormalizeModel(const NormalizeModelArgs& args);
  void RegularizeModel(const RegularizeModelArgs& args);
  void OverwriteModel(const TopicModel& args);
  int Duplicate(const DuplicateMasterComponentArgs& args);
  int ClearThetaCache(const ClearThetaCacheArgs& args);
  int ClearScoreCache(const ClearScoreCacheArgs& args);
  int ClearScoreArrayCache(const ClearScoreArrayCacheArgs& args);

  // Test helpers
  ::artm::FitOfflineMasterModelArgs Initialize(const std::vector<std::shared_ptr< ::artm::Batch> >& batches,
                                               ::artm::ImportBatchesArgs* import_batches_args = nullptr,
                                               ::artm::InitializeModelArgs* initialize_model_args = nullptr,
                                               const ::artm::DictionaryData* dictionary_data = nullptr);

 private:
  MasterModel& master_model_;
};

}  // namespace test
}  // namespace artm
