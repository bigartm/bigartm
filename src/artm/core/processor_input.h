// Copyright 2015, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_PROCESSOR_INPUT_H_
#define SRC_ARTM_CORE_PROCESSOR_INPUT_H_

#include <string>

#include "boost/uuid/uuid.hpp"

#include "artm/core/common.h"
#include "artm/messages.pb.h"

namespace artm {
namespace core {

class Notifiable {
 public:
  virtual ~Notifiable() {}
  virtual void Callback(const boost::uuids::uuid& id, const ModelName& model_name) = 0;
};

class ScoresMerger;
class CacheManager;

class ProcessorInput {
 public:
  enum Caller {
    Unknown,
    InvokeIteration,
    AddBatch,
    ProcessBatches,
  };

  ProcessorInput() : batch_(), model_config_(), model_name_(), nwt_target_name_(),
                     batch_filename_(), batch_weight_(1.0f), task_id_(),
                     notifiable_(nullptr), scores_merger_(nullptr),
                     cache_manager_(nullptr), caller_(Caller::Unknown) {}

  Batch* mutable_batch() { return &batch_; }
  const Batch& batch() const { return batch_; }

  ModelConfig* mutable_model_config() { return &model_config_; }
  const ModelConfig& model_config() const { return model_config_; }

  Notifiable* notifiable() const { return notifiable_; }
  void set_notifiable(Notifiable* notifiable) { notifiable_ = notifiable; }

  ScoresMerger* scores_merger() const { return scores_merger_; }
  void set_scores_merger(ScoresMerger* scores_merger) { scores_merger_ = scores_merger; }

  CacheManager* cache_manager() const { return cache_manager_; }
  void set_cache_manager(CacheManager* cache_manager) { cache_manager_ = cache_manager; }
  bool has_cache_manager() const { return cache_manager_ != nullptr; }

  const ModelName& model_name() const { return model_name_; }
  void set_model_name(const ModelName& model_name) { model_name_ = model_name; }

  const ModelName& nwt_target_name() const { return nwt_target_name_; }
  void set_nwt_target_name(const ModelName& nwt_target_name) { nwt_target_name_ = nwt_target_name; }
  bool has_nwt_target_name() const { return !nwt_target_name_.empty(); }

  const std::string& batch_filename() const { return batch_filename_; }
  void set_batch_filename(const std::string& batch_filename) { batch_filename_ = batch_filename; }
  bool has_batch_filename() const { return !batch_filename_.empty(); }

  const float batch_weight() const { return batch_weight_; }
  void set_batch_weight(float batch_weight) { batch_weight_ = batch_weight; }

  const boost::uuids::uuid& task_id() const { return task_id_; }
  void set_task_id(const boost::uuids::uuid& task_id) { task_id_ = task_id; }

  Caller caller() const { return caller_; }
  void set_caller(const Caller caller) { caller_ = caller; }

 private:
  Batch batch_;
  ModelConfig model_config_;
  ModelName model_name_;
  ModelName nwt_target_name_;
  std::string batch_filename_;  // if this is set batch_ is ignored;
  float batch_weight_;
  boost::uuids::uuid task_id_;
  Notifiable* notifiable_;
  ScoresMerger* scores_merger_;
  CacheManager* cache_manager_;
  Caller caller_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_PROCESSOR_INPUT_H_
