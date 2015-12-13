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

class DataLoader;
class BatchManager;
class CacheManager;
class Processor;
class Merger;
class InstanceSchema;
class Dictionary;
class DictionaryImpl;
typedef ThreadSafeCollectionHolder<std::string, Dictionary> ThreadSafeDictionaryCollection;
typedef ThreadSafeCollectionHolder<std::string, DictionaryImpl> ThreadSafeDictionaryImplCollection;
typedef ThreadSafeCollectionHolder<std::string, Batch> ThreadSafeBatchCollection;
typedef ThreadSafeQueue<std::shared_ptr<ProcessorInput>> ProcessorQueue;
typedef ThreadSafeQueue<std::shared_ptr<ModelIncrement>> MergerQueue;

// temp function to convert DictionaryData -> DictionaryConfig
std::shared_ptr<artm::DictionaryConfig> dictionary_data_to_config(std::shared_ptr<artm::DictionaryData> data,
                                                                  std::shared_ptr<artm::DictionaryData> cooc_data) {
  auto dictionary_config = std::make_shared<artm::DictionaryConfig>();
  dictionary_config->set_name(data->name());

  for (int i = 0; i < data->token_size(); ++i) {
    auto entry = dictionary_config->add_entry();
    entry->set_key_token(data->token(i));
    entry->set_class_id(data->class_id(i));
    entry->set_value(data->token_value(i));
  }

  if (cooc_data == nullptr) return dictionary_config;

  dictionary_config->clear_cooc_entries();
  auto cooc_entries = dictionary_config->mutable_cooc_entries();
  for (int i = 0; i < cooc_data->cooc_first_index_size(); ++i) {
    cooc_entries->add_first_index(cooc_data->cooc_first_index(i));
    cooc_entries->add_second_index(cooc_data->cooc_second_index(i));
    cooc_entries->add_value(cooc_data->cooc_value(i));
  }

  return dictionary_config;
}

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
  MergerQueue* merger_queue() { return &merger_queue_; }
  ThreadSafeBatchCollection* batches() { return &batches_; }

  DataLoader* data_loader();
  BatchManager* batch_manager();
  CacheManager* cache_manager();
  Merger* merger();

  int processor_size() { return processors_.size(); }
  Processor* processor(int processor_index) { return processors_[processor_index].get(); }

  void Reconfigure(const MasterComponentConfig& master_config);
  void CreateOrReconfigureModel(const ModelConfig& config);
  void DisposeModel(ModelName model_name);

  void CreateOrReconfigureRegularizer(const RegularizerConfig& config);
  void DisposeRegularizer(const std::string& name);

  void CreateOrReconfigureDictionary(const DictionaryConfig& config);
  void CreateOrReconfigureDictionaryImpl(const DictionaryData& data);
  void DisposeDictionary(const std::string& name);
  std::shared_ptr<Dictionary> dictionary(const std::string& name);
  std::shared_ptr<DictionaryImpl> dictionary_impl(const std::string& name);

  std::shared_ptr<ScoreCalculatorInterface> CreateScoreCalculator(const ScoreConfig& config);

 private:
  bool is_configured_;

  // The order of the class members defines the order in which obects are created and destroyed.
  // Pay special attantion to the order of data_loader_, merger_ and processor_,
  // because all this objects has an associated thread.
  // Such threads must be terminated prior to all the objects that the thread might potentially access.

  ThreadSafeHolder<InstanceSchema> schema_;
  ThreadSafeDictionaryCollection dictionaries_;
  ThreadSafeDictionaryImplCollection dictionaries_impl_;
  ThreadSafeBatchCollection batches_;

  ProcessorQueue processor_queue_;

  MergerQueue merger_queue_;

  // Depends on schema_
  std::shared_ptr<CacheManager> cache_manager_;

  // Depends on schema_
  std::shared_ptr<BatchManager> batch_manager_;

  // Depends on schema_, processor_queue_, batch_manager_
  std::shared_ptr<DataLoader> data_loader_;

  // Depends on schema_, merger_queue_, data_loader_
  std::shared_ptr<Merger> merger_;

  // Depends on schema_, processor_queue_, merger_queue_, and merger_
  std::vector<std::shared_ptr<Processor> > processors_;

  Instance(const Instance& rhs);
  Instance& operator=(const Instance&);
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_INSTANCE_H_
