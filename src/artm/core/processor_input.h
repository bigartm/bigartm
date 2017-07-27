// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <string>

#include "boost/uuid/uuid.hpp"

#include "artm/core/common.h"

namespace artm {
namespace core {

class BatchManager;
class ScoreManager;
class CacheManager;

// This class describes one task for the processor component.
// It has all the input data needed to execute ProcessBatch routine.
// ProcessorInput is an element of the processor queue (Instance::processor_queue_).
class ProcessorInput {
 public:
  ProcessorInput() : batch_(), args_(), model_name_(), nwt_target_name_(),
                     batch_filename_(), batch_weight_(1.0f), task_id_(), batch_manager_(nullptr),
                     score_manager_(nullptr), cache_manager_(nullptr),
                     ptdw_cache_manager_(nullptr),
                     reuse_theta_cache_manager_(nullptr) { }

  Batch* mutable_batch() { return &batch_; }
  const Batch& batch() const { return batch_; }

  ProcessBatchesArgs* mutable_args() { return &args_; }
  const ProcessBatchesArgs& args() const { return args_; }

  BatchManager* batch_manager() const { return batch_manager_; }
  void set_batch_manager(BatchManager* batch_manager) { batch_manager_ = batch_manager; }

  ScoreManager* score_manager() const { return score_manager_; }
  void set_score_manager(ScoreManager* score_manager) { score_manager_ = score_manager; }

  CacheManager* cache_manager() const { return cache_manager_; }
  void set_cache_manager(CacheManager* cache_manager) { cache_manager_ = cache_manager; }
  bool has_cache_manager() const { return cache_manager_ != nullptr; }

  CacheManager* ptdw_cache_manager() const { return ptdw_cache_manager_; }
  void set_ptdw_cache_manager(CacheManager* ptdw_cache_manager) { ptdw_cache_manager_ = ptdw_cache_manager; }
  bool has_ptdw_cache_manager() const { return ptdw_cache_manager_ != nullptr; }

  CacheManager* reuse_theta_cache_manager() const { return reuse_theta_cache_manager_; }
  void set_reuse_theta_cache_manager(CacheManager* cache_manager) { reuse_theta_cache_manager_ = cache_manager; }
  bool has_reuse_theta_cache_manager() const { return reuse_theta_cache_manager_ != nullptr; }

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

 private:
  Batch batch_;
  ProcessBatchesArgs args_;
  ModelName model_name_;
  ModelName nwt_target_name_;
  std::string batch_filename_;  // if this is set batch_ is ignored;
  float batch_weight_;
  boost::uuids::uuid task_id_;
  BatchManager* batch_manager_;
  ScoreManager* score_manager_;
  CacheManager* cache_manager_;
  CacheManager* ptdw_cache_manager_;
  CacheManager* reuse_theta_cache_manager_;
};

}  // namespace core
}  // namespace artm
