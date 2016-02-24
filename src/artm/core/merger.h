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
typedef ThreadSafeCollectionHolder<std::string, Batch> ThreadSafeBatchCollection;

class Merger : boost::noncopyable {
 public:
  Merger(ThreadSafeHolder<InstanceSchema>* schema,
         const ::artm::core::ThreadSafeBatchCollection* batches,
         const ::artm::core::ThreadSafeDictionaryCollection* dictionaries);

  ~Merger();

  void DisposeModel(ModelName model_name);
  void ResetScores(ModelName model_name);

  void OverwriteTopicModel(const ::artm::TopicModel& topic_model);
  void InitializeModel(const InitializeModelArgs& args);
  ScoresMerger* scores_merger() { return &scores_merger_; }

  std::shared_ptr<const ::artm::core::PhiMatrix> GetPhiMatrix(ModelName model_name) const;
  std::shared_ptr<const ::artm::core::PhiMatrix> GetPhiMatrixSafe(ModelName model_name) const;
  void SetPhiMatrix(ModelName model_name, std::shared_ptr< ::artm::core::PhiMatrix> phi_matrix);

  void RetrieveExternalTopicModel(const ::artm::GetTopicModelArgs& get_model_args,
                                  ::artm::TopicModel* topic_model) const;
  void RequestScore(const GetScoreValueArgs& get_score_args,
                    ScoreData *score_data) const;

  std::vector<ModelName> model_name() const;

 private:
  ThreadSafeCollectionHolder<ModelName, PhiMatrix> phi_matrix_;
  ThreadSafeHolder<InstanceSchema>* schema_;
  ScoresMerger scores_merger_;

  const ::artm::core::ThreadSafeBatchCollection* batches_;
  const ::artm::core::ThreadSafeDictionaryCollection* dictionaries_;
};

}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_MERGER_H_
