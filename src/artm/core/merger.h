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

#include "rpcz/sync_event.hpp"

#include "artm/core/common.h"
#include "artm/core/internals.rpcz.h"
#include "artm/core/thread_safe_holder.h"
#include "artm/score_calculator_interface.h"

namespace artm {
namespace core {

class TopicModel;
class InstanceSchema;

class Merger : boost::noncopyable {
 public:
  Merger(ThreadSafeQueue<std::shared_ptr<const ModelIncrement> >* merger_queue,
         ThreadSafeHolder<InstanceSchema>* schema,
         MasterComponentService_Stub* master_component_service,
         Notifiable* notifiable);

  ~Merger();

  void DisposeModel(ModelName model_name);
  void CreateOrReconfigureModel(const ModelConfig& model);
  void ForceResetScores(ModelName model_name);

  // Returns false if BigARTM is still processing the collection, otherwise true.
  bool WaitIdle(int timeout = -1);
  void ForcePullTopicModel();
  void ForcePushTopicModelIncrement();
  void InvokePhiRegularizers();
  void OverwriteTopicModel(const ::artm::TopicModel& topic_model);

  std::shared_ptr<const ::artm::core::TopicModel> GetLatestTopicModel(ModelName model_name) const;
  bool RetrieveExternalTopicModel(ModelName model_name, ::artm::TopicModel* topic_model) const;
  void RequestRegularizerState(RegularizerName regularizer_name,
                               ::artm::RegularizerInternalState* regularizer_state) const;
  bool RequestScore(const ModelName& model_name, const ScoreName& score_name,
                    ScoreData *score_data) const;

 private:
  class ScoresMerger {
   public:
    explicit ScoresMerger(ThreadSafeHolder<InstanceSchema>* schema,
                          ThreadSafeCollectionHolder<ModelName, TopicModel>* topic_model)
        : schema_(schema), topic_model_(topic_model), score_map_() {}

    void Append(const ModelName& model_name, const ScoreName& score_name,
                const std::string& score_blob);

    void ResetScores(const ModelName& model_name);
    void RetrieveModelIncrement(const ModelName& model_name, ModelIncrement* model_increment);
    bool RequestScore(const ModelName& model_name, const ScoreName& score_name,
                      ScoreData *score_data) const;

   private:
    ThreadSafeHolder<InstanceSchema>* schema_;
    ThreadSafeCollectionHolder<ModelName, TopicModel>* topic_model_;

    // Map from model name and score name to the score
    typedef std::pair<ModelName, ScoreName> ScoreKey;
    ThreadSafeCollectionHolder<ScoreKey, Score> score_map_;
  };

  enum MergerTaskType {
    kDisposeModel,
    kForcePullTopicModel,
    kForcePushTopicModelIncrement,
    kForceResetScores,
  };

  struct MergerTask {
    MergerTask() {}
    MergerTask(MergerTaskType _task_type, ModelName _model_name)
        : task_type(_task_type), model_name(_model_name), sync_event(nullptr) {}

    MergerTask(MergerTaskType _task_type, ModelName _model_name, rpcz::sync_event* _sync_event)
        : task_type(_task_type), model_name(_model_name), sync_event(_sync_event) {}

    MergerTaskType task_type;
    ModelName model_name;
    rpcz::sync_event* sync_event;
  };

  ThreadSafeCollectionHolder<ModelName, TopicModel> topic_model_;
  std::map<ModelName, std::shared_ptr<TopicModel>> topic_model_inc_;
  ThreadSafeHolder<InstanceSchema>* schema_;
  artm::core::MasterComponentService_Stub* master_component_service_;
  ScoresMerger scores_merger_;

  mutable std::atomic<bool> is_idle_;
  ThreadSafeQueue<std::shared_ptr<const ModelIncrement> >* merger_queue_;
  ThreadSafeQueue<MergerTask> internal_task_queue_;

  Notifiable* notifiable_;

  mutable std::atomic<bool> is_stopping;
  boost::thread thread_;
  void ThreadFunction();

  void PullTopicModel();
  void PushTopicModelIncrement();
  void ResetScores(ModelName model_name);
};

}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_MERGER_H_
