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

class ProcessorInput {
 public:
  ProcessorInput() : batch_(), model_config_(), model_name_(), batch_filename_(), task_id_(),
                     notifiable_(nullptr) {}

  Batch* mutable_batch() { return &batch_; }
  const Batch& batch() const { return batch_; }

  ModelConfig* mutable_model_config() { return &model_config_; }
  const ModelConfig& model_config() const { return model_config_; }

  Notifiable* notifiable() const { return notifiable_; }
  void set_notifiable(Notifiable* notifiable) { notifiable_ = notifiable; }

  const ModelName& model_name() const { return model_name_; }
  void set_model_name(const ModelName& model_name) { model_name_ = model_name; }

  const std::string& batch_filename() const { return batch_filename_; }
  void set_batch_filename(const std::string& batch_filename) { batch_filename_ = batch_filename; }
  bool has_batch_filename() const { return !batch_filename_.empty(); }

  const boost::uuids::uuid& task_id() const { return task_id_; }
  void set_task_id(const boost::uuids::uuid& task_id) { task_id_ = task_id; }

 private:
  Batch batch_;
  ModelConfig model_config_;
  ModelName model_name_;
  std::string batch_filename_;  // if this is set batch_ is ignored;
  boost::uuids::uuid task_id_;
  Notifiable* notifiable_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_PROCESSOR_INPUT_H_
