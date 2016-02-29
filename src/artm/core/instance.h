// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_INSTANCE_H_
#define SRC_ARTM_CORE_INSTANCE_H_

#include <map>
#include <memory>
#include <vector>
#include <string>

#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"

#include "artm/messages.pb.h"

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"
#include "artm/core/processor_input.h"
#include "artm/core/thread_safe_holder.h"

#include "artm/score_calculator_interface.h"

namespace artm {
namespace core {

class BatchManager;
class CacheManager;
class ScoreManager;
class ScoreTracker;
class Processor;
class Merger;
class InstanceSchema;
class Dictionary;
typedef ThreadSafeCollectionHolder<std::string, Dictionary> ThreadSafeDictionaryCollection;
typedef ThreadSafeCollectionHolder<std::string, Batch> ThreadSafeBatchCollection;
typedef ThreadSafeCollectionHolder<std::string, PhiMatrix> ThreadSafeModelCollection;
typedef ThreadSafeQueue<std::shared_ptr<ProcessorInput>> ProcessorQueue;

// Class Instance is respondible for joint hosting of many other components
// (processors, merger, data loader) and data structures (schema, queues, etc).
class Instance {
 public:
  explicit Instance(const MasterComponentConfig& config);
  ~Instance();

  std::shared_ptr<Instance> Duplicate() const;
  void RequestMasterComponentInfo(MasterComponentInfo* master_info) const;

  std::shared_ptr<InstanceSchema> schema() const { return schema_.get(); }
  ProcessorQueue* processor_queue() { return &processor_queue_; }
  ThreadSafeDictionaryCollection* dictionaries() { return &dictionaries_; }
  ThreadSafeBatchCollection* batches() { return &batches_; }
  ThreadSafeModelCollection* models() { return &models_; }

  BatchManager* batch_manager();
  CacheManager* cache_manager();
  ScoreManager* score_manager();
  ScoreTracker* score_tracker();

  int processor_size() { return processors_.size(); }
  Processor* processor(int processor_index) { return processors_[processor_index].get(); }

  void Reconfigure(const MasterComponentConfig& master_config);
  void DisposeModel(ModelName model_name);

  void CreateOrReconfigureRegularizer(const RegularizerConfig& config);
  void DisposeRegularizer(const std::string& name);

  std::shared_ptr<ScoreCalculatorInterface> CreateScoreCalculator(const ScoreConfig& config);

  std::shared_ptr<const ::artm::core::PhiMatrix> GetPhiMatrix(ModelName model_name) const;
  std::shared_ptr<const ::artm::core::PhiMatrix> GetPhiMatrixSafe(ModelName model_name) const;
  void SetPhiMatrix(ModelName model_name, std::shared_ptr< ::artm::core::PhiMatrix> phi_matrix);

 private:
  bool is_configured_;

  // The order of the class members defines the order in which obects are created and destroyed.
  // Pay special attantion to the location of processor_,
  // because it has an associated thread.
  // Such threads must be terminated prior to all the objects that the thread might potentially access.

  ThreadSafeHolder<InstanceSchema> schema_;
  ThreadSafeDictionaryCollection dictionaries_;
  ThreadSafeBatchCollection batches_;
  ThreadSafeModelCollection models_;

  ProcessorQueue processor_queue_;

  // Depends on schema_
  std::shared_ptr<CacheManager> cache_manager_;

  // Depends on schema_
  std::shared_ptr<BatchManager> batch_manager_;

  // Depends on [none]
  std::shared_ptr<ScoreManager> score_manager_;
  std::shared_ptr<ScoreTracker> score_tracker_;

  // Depends on schema_, processor_queue_, and merger_
  std::vector<std::shared_ptr<Processor> > processors_;

  Instance(const Instance& rhs);
  Instance& operator=(const Instance&);
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_INSTANCE_H_
