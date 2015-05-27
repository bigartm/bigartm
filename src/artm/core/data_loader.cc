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
#include "artm/core/processor_input.h"
#include "artm/core/helpers.h"
#include "artm/core/merger.h"

namespace fs = boost::filesystem;

namespace artm {
namespace core {

Instance* DataLoader::instance() {
  return instance_;
}

DataLoader::DataLoader(Instance* instance) : instance_(instance) {}

DataLoader::~DataLoader() {}

bool DataLoader::AddBatch(const AddBatchArgs& args) {
  if (!args.has_batch() && !args.has_batch_file_name()) {
    std::string message = "AddBatchArgs.batch or AddBatchArgs.batch_file_name must be specified";
    BOOST_THROW_EXCEPTION(InvalidOperation(message));
  }

  int timeout = args.timeout_milliseconds();
  std::shared_ptr<InstanceSchema> schema = instance()->schema();
  const MasterComponentConfig& config = schema->config();

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

  std::vector<ModelName> model_names = schema->GetModelNames();
  std::for_each(model_names.begin(), model_names.end(), [&](ModelName model_name) {
    if (!schema->has_model_config(model_name))
      return;  // return from lambda and continues for_each

    boost::uuids::uuid task_id = boost::uuids::random_generator()();
    instance_->batch_manager()->Add(task_id, std::string(), model_name);

    auto pi = std::make_shared<ProcessorInput>();
    pi->set_notifiable(instance_->batch_manager());
    pi->set_model_name(model_name);
    pi->mutable_batch()->CopyFrom(*batch);
    pi->mutable_model_config()->CopyFrom(schema->model_config(model_name));
    pi->set_task_id(task_id);
    instance_->processor_queue()->push(pi);
  });

  return true;
}

void DataLoader::InvokeIteration(const InvokeIterationArgs& args) {
  int iterations_count = args.iterations_count();
  if (iterations_count <= 0) {
    LOG(WARNING) << "DataLoader::InvokeIteration() was called with argument '"
                 << iterations_count << "'. Call is ignored.";
    return;
  }

  std::string disk_path = args.has_disk_path() ? args.disk_path() : instance()->schema()->config().disk_path();
  std::vector<std::string> tasks = BatchHelpers::ListAllBatches(disk_path);

  if (tasks.empty()) {
    LOG(WARNING) << "DataLoader::InvokeIteration() - no batches found, "
                 << "please populate DataLoader data with some data";
    return;
  }

  std::shared_ptr<InstanceSchema> schema = instance()->schema();
  std::vector<ModelName> model_names = schema->GetModelNames();

  for (int iter = 0; iter < iterations_count; ++iter) {
    for (const std::string& task : tasks) {
      std::for_each(model_names.begin(), model_names.end(), [&](ModelName model_name) {
        if (!schema->has_model_config(model_name))
          return;  // return from lambda and continues for_each

        boost::uuids::uuid task_id = boost::uuids::random_generator()();
        instance_->batch_manager()->Add(task_id, task, model_name);

        std::shared_ptr<ProcessorInput> pi = std::make_shared<ProcessorInput>();
        pi->set_notifiable(instance()->batch_manager());
        pi->set_task_id(task_id);
        pi->set_batch_filename(task);
        pi->set_model_name(model_name);
        pi->mutable_model_config()->CopyFrom(schema->model_config(model_name));
        instance()->processor_queue()->push(pi);
      });
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

}  // namespace core
}  // namespace artm
