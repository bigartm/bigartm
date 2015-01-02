// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_PROCESSOR_H_
#define SRC_ARTM_CORE_PROCESSOR_H_

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/bind.hpp"
#include "boost/utility.hpp"

#include "artm/messages.pb.h"

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"
#include "artm/core/thread_safe_holder.h"

#include "artm/utility/blas.h"

namespace artm {
namespace core {

class InstanceSchema;
class Merger;
class TopicModel;
class TopicWeightIterator;

class Processor : boost::noncopyable {
 public:
  Processor(ThreadSafeQueue<std::shared_ptr<ProcessorInput> >*  processor_queue,
            ThreadSafeQueue<std::shared_ptr<ModelIncrement> >* merger_queue,
            const Merger& merger,
            const ThreadSafeHolder<InstanceSchema>& schema);

  ~Processor();

  void FindThetaMatrix(const Batch& batch, const GetThetaMatrixArgs& args, ThetaMatrix* theta_matrix);

 private:
  ThreadSafeQueue<std::shared_ptr<ProcessorInput> >* processor_queue_;
  ThreadSafeQueue<std::shared_ptr<ModelIncrement> >* merger_queue_;
  const Merger& merger_;
  const ThreadSafeHolder<InstanceSchema>& schema_;

  mutable std::atomic<bool> is_stopping;
  boost::thread thread_;

  void ThreadFunction();

  // Helper class to iterate on stream
  class StreamIterator : boost::noncopyable {
   public:
    // Iterates on a global stream (all items in the batch)
    explicit StreamIterator(const ProcessorInput& processor_input);

    const Item* Next();
    const Item* Current() const;

    // Checks whether Current() item belongs to a specific stream
    bool InStream(const std::string& stream_name);
    bool InStream(int stream_index);

    inline int item_index() const { return item_index_; }

   private:
    int items_count_;
    int item_index_;
    const Mask* stream_flags_;
    const ProcessorInput& processor_input_;
  };
};

}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_PROCESSOR_H_
