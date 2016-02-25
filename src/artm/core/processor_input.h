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

class ScoreManager;
class CacheManager;

class ProcessorInput {
 public:
  ProcessorInput() : batch_(), model_config_(), model_name_(), nwt_target_name_(),
                     batch_filename_(), batch_weight_(1.0f), task_id_(), notifiable_(nullptr),
                     score_manager_(nullptr), cache_manager_(nullptr),
                     ptdw_cache_manager_(nullptr),
                     reuse_theta_cache_manager_(nullptr) {}

  Batch* mutable_batch() { return &batch_; }
  const Batch& batch() const { return batch_; }

  ModelConfig* mutable_model_config() { return &model_config_; }
  const ModelConfig& model_config() const { return model_config_; }

  Notifiable* notifiable() const { return notifiable_; }
  void set_notifiable(Notifiable* notifiable) { notifiable_ = notifiable; }

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
  ModelConfig model_config_;
  ModelName model_name_;
  ModelName nwt_target_name_;
  std::string batch_filename_;  // if this is set batch_ is ignored;
  float batch_weight_;
  boost::uuids::uuid task_id_;
  Notifiable* notifiable_;
  ScoreManager* score_manager_;
  CacheManager* cache_manager_;
  CacheManager* ptdw_cache_manager_;
  CacheManager* reuse_theta_cache_manager_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_PROCESSOR_INPUT_H_
