// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_INSTANCE_H_
#define SRC_ARTM_CORE_INSTANCE_H_

#include <map>
#include <memory>
#include <vector>
#include <string>

#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"

#include "rpcz/application.hpp"

#include "artm/messages.pb.h"

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"
#include "artm/core/internals.rpcz.h"
#include "artm/core/template_manager.h"
#include "artm/core/thread_safe_holder.h"

#include "artm/score_calculator_interface.h"

namespace artm {
namespace core {

class LocalDataLoader;
class RemoteDataLoader;
class BatchManager;
class Processor;
class Merger;
class InstanceSchema;
typedef std::map<Token, ::artm::DictionaryEntry> DictionaryMap;
typedef ThreadSafeCollectionHolder<std::string, DictionaryMap> ThreadSafeDictionaryCollection;
typedef ThreadSafeQueue<std::shared_ptr<const ProcessorInput>> ProcessorQueue;
typedef ThreadSafeQueue<std::shared_ptr<const ModelIncrement>> MergerQueue;

// Instance type defines which components will be hosted in the instance.
// =============================
// The following components are always hosted:
//   - dictionaries   - yes
//   - regularizers   - yes
//   - schema         - yes
// =============================
enum InstanceType {
  // - merger        - yes, master
  // - processors    - yes
  // - batch_manager - yes
  // - data_loader   - yes, local
  MasterInstanceLocal,

  // - merger        - yes, master
  // - processor     - no
  // - batch_manager - yes
  // - data_loader   - no
  MasterInstanceNetwork,

  // - merger        - yes, working
  // - processors    - yes
  // - batch_manager - no
  // - data_loader   - yes, remote
  NodeControllerInstance,
};

// Class Instance is respondible for joint hosting of many other components
// (processors, merger, data loader) and data structures (schema, queues, etc).
// The set of objects, hosted in the Instance, depends on instance_type.
class Instance : boost::noncopyable {
 public:
  explicit Instance(const MasterComponentConfig& config, InstanceType instance_type);
  ~Instance();

  InstanceType type() const { return instance_type_; }

  std::shared_ptr<InstanceSchema> schema() const { return schema_.get(); }
  ProcessorQueue* processor_queue() { return &processor_queue_; }
  MergerQueue* merger_queue() { return &merger_queue_; }

  LocalDataLoader* local_data_loader();
  bool has_local_data_loader() { return local_data_loader_ != nullptr; }

  RemoteDataLoader* remote_data_loader();
  bool has_remote_data_loader() { return remote_data_loader_ != nullptr; }

  BatchManager* batch_manager();
  bool has_batch_manager() { return batch_manager_ != nullptr; }

  MasterComponentService_Stub* master_component_service_proxy();
  bool has_master_component_service_proxy() { return master_component_service_proxy_ != nullptr; }

  Merger* merger();
  bool has_merger() { return merger_ != nullptr; }

  void Reconfigure(const MasterComponentConfig& master_config);
  void CreateOrReconfigureModel(const ModelConfig& config);
  void DisposeModel(ModelName model_name);
  void CreateOrReconfigureRegularizer(const RegularizerConfig& config);
  void DisposeRegularizer(const std::string& name);
  void CreateOrReconfigureDictionary(const DictionaryConfig& config);
  void DisposeDictionary(const std::string& name);

  std::shared_ptr<ScoreCalculatorInterface> CreateScoreCalculator(const ScoreConfig& config);

 private:
  void PopulateClassId(ModelConfig* model_config);

  bool is_configured_;
  InstanceType instance_type_;

  // The order of the class members defines the order in which obects are created and destroyed.
  // Pay special attantion to the order of data_loader_, merger_ and processor_,
  // because all this objects has an associated thread.
  // Such threads must be terminated prior to all the objects that the thread might potentially access.

  ThreadSafeHolder<InstanceSchema> schema_;
  ThreadSafeDictionaryCollection dictionaries_;

  std::unique_ptr<rpcz::application> application_;

  // Depends on application_
  std::shared_ptr<artm::core::MasterComponentService_Stub> master_component_service_proxy_;

  ProcessorQueue processor_queue_;

  MergerQueue merger_queue_;

  // Depends on schema_
  std::shared_ptr<BatchManager> batch_manager_;

  // Depends on schema_, master_component_service_proxy_, processor_queue_, batch_manager_
  std::shared_ptr<LocalDataLoader> local_data_loader_;
  std::shared_ptr<RemoteDataLoader> remote_data_loader_;

  // Depends on schema_, master_component_service_proxy_, merger_queue_, local&remote_data_loader_
  std::shared_ptr<Merger> merger_;

  // Depends on schema_, processor_queue_, merger_queue_, and merger_
  std::vector<std::shared_ptr<Processor> > processors_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_INSTANCE_H_
