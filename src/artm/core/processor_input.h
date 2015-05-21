// Copyright 2015, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_PROCESSOR_INPUT_H_
#define SRC_ARTM_CORE_PROCESSOR_INPUT_H_

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
  ProcessorInput() : notifiable_(nullptr) {}
  Batch* mutable_batch() { return &batch_; }
  const Batch& batch() const { return batch_; }
  Notifiable* notifiable() const { return notifiable_; }
  void set_notifiable(Notifiable* notifiable) { notifiable_ = notifiable; }

 private:
  Batch batch_;
  Notifiable* notifiable_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_PROCESSOR_INPUT_H_
