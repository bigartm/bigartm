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
#include "artm/core/processor_input.h"
#include "artm/core/thread_safe_holder.h"

#include "artm/utility/blas.h"

namespace artm {
namespace core {

class InstanceSchema;
class Merger;
class CacheManager;
class TopicModel;

class Processor : boost::noncopyable {
 public:
  Processor(ThreadSafeQueue<std::shared_ptr<ProcessorInput> >* processor_queue,
            ThreadSafeQueue<std::shared_ptr<ModelIncrement> >* merger_queue,
            const Merger& merger,
            const CacheManager& cache_manager,
            const ThreadSafeHolder<InstanceSchema>& schema);

  ~Processor();

  void FindThetaMatrix(const Batch& batch,
                       const GetThetaMatrixArgs& args, ThetaMatrix* theta_matrix,
                       const GetScoreValueArgs& score_args, ScoreData* score_result);

 private:
  ThreadSafeQueue<std::shared_ptr<ProcessorInput> >* processor_queue_;
  ThreadSafeQueue<std::shared_ptr<ModelIncrement> >* merger_queue_;
  const Merger& merger_;
  const CacheManager& cache_manager_;
  const ThreadSafeHolder<InstanceSchema>& schema_;

  mutable std::atomic<bool> is_stopping;
  boost::thread thread_;

  void ThreadFunction();
};

}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_PROCESSOR_H_
