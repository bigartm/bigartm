// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/data_loader.h"

#include <string>
#include <vector>
#include <fstream>  // NOLINT

#include "boost/exception/diagnostic_information.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

#include "glog/logging.h"

#include "artm/core/exceptions.h"
#include "artm/core/instance.h"
#include "artm/core/batch_manager.h"
#include "artm/core/instance_schema.h"
#include "artm/core/protobuf_helpers.h"
#include "artm/core/helpers.h"
#include "artm/core/generation.h"
#include "artm/core/merger.h"

namespace fs = boost::filesystem;

namespace artm {
namespace core {

Instance* DataLoader::instance() {
  return instance_;
}

DataLoader::DataLoader(Instance* instance)
    : instance_(instance),
      generation_(nullptr),
      is_stopping(false),
      thread_() {
  std::string disk_path = instance->schema()->config().disk_path();
  if (!disk_path.empty()) {
    generation_.reset(new DiskGeneration(disk_path));
  }

  // Keep this at the last action in constructor.
  // http://stackoverflow.com/questions/15751618/initialize-boost-thread-in-object-constructor
  boost::thread t(&DataLoader::ThreadFunction, this);
  thread_.swap(t);
}

DataLoader::~DataLoader() {
  is_stopping = true;
  if (thread_.joinable()) {
    thread_.join();
  }
}

bool DataLoader::AddBatch(const AddBatchArgs& args) {
  if (!args.has_batch() && !args.has_batch_file_name()) {
    std::string message = "AddBatchArgs.batch or AddBatchArgs.batch_file_name must be specified";
    BOOST_THROW_EXCEPTION(InvalidOperation(message));
  }

  int timeout = args.timeout_milliseconds();
  MasterComponentConfig config = instance()->schema()->config();

  std::shared_ptr<Batch> batch = std::make_shared< ::artm::Batch>();
  if (args.has_batch_file_name()) {
    ::artm::core::BatchHelpers::LoadMessage(args.batch_file_name(), batch.get());
  } else {
    batch = std::make_shared<Batch>(args.batch());  // copy constructor
  }

  if (config.compact_batches()) {
    std::shared_ptr<Batch> modified_batch = std::make_shared<Batch>();  // constructor
    BatchHelpers::CompactBatch(*batch, modified_batch.get());
    batch = modified_batch;
  }

  auto time_start = boost::posix_time::microsec_clock::local_time();
  for (;;) {
    if (instance_->processor_queue()->size() < config.processor_queue_max_size()) break;

    boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));

    if (timeout >= 0) {
      auto time_end = boost::posix_time::microsec_clock::local_time();
      if ((time_end - time_start).total_milliseconds() >= timeout) return false;
    }
  }
  auto pi = std::make_shared<ProcessorInput>();
  pi->mutable_batch()->CopyFrom(*batch);
  pi->set_batch_uuid(batch->id());
  boost::uuids::uuid uuid = boost::lexical_cast<boost::uuids::uuid>(batch->id());
  instance_->batch_manager()->AddAndNext(BatchManagerTask(uuid, std::string()));
  instance_->processor_queue()->push(pi);

  return true;
}

void DataLoader::InvokeIteration(const InvokeIterationArgs& args) {
  int iterations_count = args.iterations_count();
  if (iterations_count <= 0) {
    LOG(WARNING) << "DataLoader::InvokeIteration() was called with argument '"
                 << iterations_count << "'. Call is ignored.";
    return;
  }

  DiskGeneration* generation;
  std::unique_ptr<DiskGeneration> args_generation;
  if (args.has_disk_path()) {
    args_generation.reset(new DiskGeneration(args.disk_path()));
    generation = args_generation.get();
  } else {
    generation = generation_.get();
  }

  if (generation == nullptr || generation->empty()) {
    LOG(WARNING) << "DataLoader::InvokeIteration() - current generation is empty, "
                 << "please populate DataLoader data with some data";
    return;
  }

  std::vector<BatchManagerTask> tasks = generation->batch_uuids();
  for (int iter = 0; iter < iterations_count; ++iter) {
    for (auto &task : tasks) {
      instance_->batch_manager()->Add(task);
    }
  }
}

bool DataLoader::WaitIdle(const WaitIdleArgs& args) {
  int timeout = args.timeout_milliseconds();
  auto time_start = boost::posix_time::microsec_clock::local_time();
  for (;;) {
    if (instance_->batch_manager()->IsEverythingProcessed())
      break;

    boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));

    if (timeout >= 0) {
      auto time_end = boost::posix_time::microsec_clock::local_time();
      if ((time_end - time_start).total_milliseconds() >= timeout) return false;
    }
  }

  return true;
}

void DataLoader::Callback(ModelIncrement* model_increment) {
  instance_->batch_manager()->Callback(model_increment);
}

void DataLoader::ThreadFunction() {
  try {
    Helpers::SetThreadName(-1, "DataLoader thread");
    LOG(INFO) << "DataLoader thread started";
    for (;;) {
      if (is_stopping) {
        LOG(INFO) << "DataLoader thread stopped";
        break;
      }

      auto schema = instance()->schema();
      auto config = schema->config();

      if (instance()->processor_queue()->size() >= config.processor_queue_max_size()) {
        boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));
        continue;
      }

      CuckooWatch cuckoo("LoadBatch", 2);

      BatchManagerTask next_task = instance_->batch_manager()->Next();
      if (next_task.uuid.is_nil()) {
        boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));
        continue;
      }

      std::shared_ptr<ProcessorInput> pi = std::make_shared<ProcessorInput>();
      try {
        CuckooWatch cuckoo2(std::string("LoadMessage(") + next_task.file_path + ")", &cuckoo);
        ::artm::core::BatchHelpers::LoadMessage(next_task.file_path, pi->mutable_batch());

        // keep batch.id and task.uuid in sync
        pi->set_batch_uuid(boost::lexical_cast<std::string>(next_task.uuid));
        pi->mutable_batch()->set_id(boost::lexical_cast<std::string>(next_task.uuid));
      } catch (std::exception& ex) {
        LOG(ERROR) << ex.what() << ", the batch will be skipped.";
        pi = nullptr;
      }

      if (pi == nullptr) {
        instance_->batch_manager()->Done(next_task.uuid, ModelName());
        continue;
      }

      instance()->processor_queue()->push(pi);
    }
  } catch(...) {
    LOG(FATAL) << boost::current_exception_diagnostic_information();
  }
}

}  // namespace core
}  // namespace artm
