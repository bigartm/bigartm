// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_MERGER_H_
#define SRC_ARTM_CORE_MERGER_H_

#include <assert.h>
#include <stdlib.h>

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "boost/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"

#include "artm/core/common.h"
#include "artm/core/phi_matrix.h"
#include "artm/core/dictionary.h"
#include "artm/core/sync_event.h"
#include "artm/core/scores_merger.h"
#include "artm/core/thread_safe_holder.h"
#include "artm/score_calculator_interface.h"

namespace artm {
namespace core {

class TopicModel;
class InstanceSchema;
class Dictionary;
class DictionaryImpl;
typedef ThreadSafeCollectionHolder<std::string, Batch> ThreadSafeBatchCollection;

class Merger : boost::noncopyable {
 public:
  Merger(ThreadSafeQueue<std::shared_ptr<ModelIncrement> >* merger_queue,
         ThreadSafeHolder<InstanceSchema>* schema,
         const ::artm::core::ThreadSafeBatchCollection* batches,
         const ::artm::core::ThreadSafeDictionaryCollection* dictionaries,
         const ::artm::core::ThreadSafeDictionaryImplCollection* dictionaries_impl);

  ~Merger();

  void DisposeModel(ModelName model_name);
  void CreateOrReconfigureModel(const ModelConfig& model);
  void ForceResetScores(ModelName model_name);

  // Returns false if BigARTM is still processing the collection, otherwise true.
  bool WaitIdle(const WaitIdleArgs& args);
  void ForceSynchronizeModel(const SynchronizeModelArgs& args);
  void OverwriteTopicModel(const ::artm::TopicModel& topic_model);
  void InitializeModel(const InitializeModelArgs& args);
  ScoresMerger* scores_merger() { return &scores_merger_; }

  std::shared_ptr<const ::artm::core::TopicModel> GetLatestTopicModel(ModelName model_name) const;
  std::shared_ptr<const ::artm::core::PhiMatrix> GetPhiMatrix(ModelName model_name) const;
  void SetPhiMatrix(ModelName model_name, std::shared_ptr< ::artm::core::PhiMatrix> phi_matrix);

  bool RetrieveExternalTopicModel(const ::artm::GetTopicModelArgs& get_model_args,
                                  ::artm::TopicModel* topic_model) const;
  void RequestRegularizerState(RegularizerName regularizer_name,
                               ::artm::RegularizerInternalState* regularizer_state) const;
  bool RequestScore(const GetScoreValueArgs& get_score_args,
                    ScoreData *score_data) const;
  void RequestDictionary(const DictionaryName& dictionary_name, DictionaryData* dictionary_data) const;

  std::vector<ModelName> model_name() const;

 private:
  enum MergerTaskType {
    kDisposeModel,
    kForceSynchronizeTopicModel,
    kForceResetScores,
  };

  struct MergerTask {
    MergerTask() {}

    MergerTask(MergerTaskType _task_type, ModelName _model_name, float _decay_weight,
               float _apply_weight, bool _invoke_regularizers, SyncEvent* _sync_event)
        : task_type(_task_type), model_name(_model_name), decay_weight(_decay_weight),
          apply_weight(_apply_weight), invoke_regularizers(_invoke_regularizers),
          sync_event(_sync_event) {}

    MergerTaskType task_type;
    ModelName model_name;
    float decay_weight;
    float apply_weight;
    bool invoke_regularizers;
    SyncEvent* sync_event;
  };

  ThreadSafeCollectionHolder<ModelName, TopicModel> topic_model_;
  std::map<ModelName, std::shared_ptr<TopicModel>> topic_model_inc_;
  ThreadSafeCollectionHolder<ModelName, PhiMatrix> phi_matrix_;
  ThreadSafeHolder<InstanceSchema>* schema_;
  ThreadSafeCollectionHolder<ModelName, artm::ModelConfig> target_model_config_;
  ScoresMerger scores_merger_;

  mutable std::atomic<bool> is_idle_;
  ThreadSafeQueue<std::shared_ptr<ModelIncrement> >* merger_queue_;
  ThreadSafeQueue<MergerTask> internal_task_queue_;

  const ::artm::core::ThreadSafeBatchCollection* batches_;
  const ::artm::core::ThreadSafeDictionaryCollection* dictionaries_;
  const ::artm::core::ThreadSafeDictionaryImplCollection* dictionaries_impl_;

  mutable std::atomic<bool> is_stopping;
  boost::thread thread_;
  void ThreadFunction();

  void SynchronizeModel(const ModelName& model_name, float decay_weight, float apply_weight,
                        bool invoke_regularizers);
  void ResetScores(ModelName model_name);
};

}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_MERGER_H_
