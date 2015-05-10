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
#include "artm/core/template_manager.h"
#include "artm/core/thread_safe_holder.h"

#include "artm/score_calculator_interface.h"

namespace artm {
namespace core {

class DataLoader;
class BatchManager;
class Processor;
class Merger;
class InstanceSchema;
class Dictionary;
typedef ThreadSafeCollectionHolder<std::string, Dictionary> ThreadSafeDictionaryCollection;
typedef ThreadSafeQueue<std::shared_ptr<ProcessorInput>> ProcessorQueue;
typedef ThreadSafeQueue<std::shared_ptr<ModelIncrement>> MergerQueue;

// Class Instance is respondible for joint hosting of many other components
// (processors, merger, data loader) and data structures (schema, queues, etc).
class Instance : boost::noncopyable {
 public:
  explicit Instance(const MasterComponentConfig& config);
  ~Instance();

  std::shared_ptr<InstanceSchema> schema() const { return schema_.get(); }
  ProcessorQueue* processor_queue() { return &processor_queue_; }
  MergerQueue* merger_queue() { return &merger_queue_; }

  DataLoader* data_loader();
  BatchManager* batch_manager();
  Merger* merger();

  int processor_size() { return processors_.size(); }
  Processor* processor(int processor_index) { return processors_[processor_index].get(); }

  void Reconfigure(const MasterComponentConfig& master_config);
  void CreateOrReconfigureModel(const ModelConfig& config);
  void DisposeModel(ModelName model_name);
  void CreateOrReconfigureRegularizer(const RegularizerConfig& config);
  void DisposeRegularizer(const std::string& name);
  void CreateOrReconfigureDictionary(const DictionaryConfig& config);
  void DisposeDictionary(const std::string& name);

  std::shared_ptr<ScoreCalculatorInterface> CreateScoreCalculator(const ScoreConfig& config);

 private:
  bool is_configured_;

  // The order of the class members defines the order in which obects are created and destroyed.
  // Pay special attantion to the order of data_loader_, merger_ and processor_,
  // because all this objects has an associated thread.
  // Such threads must be terminated prior to all the objects that the thread might potentially access.

  ThreadSafeHolder<InstanceSchema> schema_;
  ThreadSafeDictionaryCollection dictionaries_;

  ProcessorQueue processor_queue_;

  MergerQueue merger_queue_;

  // Depends on schema_
  std::shared_ptr<BatchManager> batch_manager_;

  // Depends on schema_, processor_queue_, batch_manager_
  std::shared_ptr<DataLoader> data_loader_;

  // Depends on schema_, merger_queue_, data_loader_
  std::shared_ptr<Merger> merger_;

  // Depends on schema_, processor_queue_, merger_queue_, and merger_
  std::vector<std::shared_ptr<Processor> > processors_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_INSTANCE_H_
