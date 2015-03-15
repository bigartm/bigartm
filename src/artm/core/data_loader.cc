// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/data_loader.h"

#include <string>
#include <vector>
#include <fstream>  // NOLINT

#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

#include "glog/logging.h"

#include "rpcz/application.hpp"

#include "artm/core/exceptions.h"
#include "artm/core/instance.h"
#include "artm/core/batch_manager.h"
#include "artm/core/instance_schema.h"
#include "artm/core/internals.rpcz.h"
#include "artm/core/protobuf_helpers.h"
#include "artm/core/helpers.h"
#include "artm/core/zmq_context.h"
#include "artm/core/generation.h"
#include "artm/core/merger.h"

namespace fs = boost::filesystem;

namespace artm {
namespace core {

DataLoader::DataLoader(Instance* instance)
    : instance_(instance) { }

Instance* DataLoader::instance() {
  return instance_;
}

void DataLoader::PopulateDataStreams(const Batch& batch, ProcessorInput* pi) {
  // loop through all streams
  MasterComponentConfig config = instance()->schema()->config();
  for (int stream_index = 0; stream_index < config.stream_size(); ++stream_index) {
    const Stream& stream = config.stream(stream_index);
    pi->add_stream_name(stream.name());

    Mask* mask = pi->add_stream_mask();
    for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
      // verify if item is part of the stream
      bool value = false;
      switch (stream.type()) {
        case Stream_Type_Global: {
          value = true;
          break;  // Stream_Type_Global
        }

        case Stream_Type_ItemIdModulus: {
          int id_mod = batch.item(item_index).id() % stream.modulus();
          value = repeated_field_contains(stream.residuals(), id_mod);
          break;  // Stream_Type_ItemIdModulus
        }

        default:
          BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException("stream.type", stream.type()));
      }

      mask->add_value(value);
    }
  }
}

LocalDataLoader::LocalDataLoader(Instance* instance)
    : DataLoader(instance),
      generation_(nullptr),
      cache_(),
      is_stopping(false),
      thread_() {
  std::string disk_path = instance->schema()->config().disk_path();
  if (!disk_path.empty()) {
    generation_.reset(new DiskGeneration(disk_path));
  }

  // Keep this at the last action in constructor.
  // http://stackoverflow.com/questions/15751618/initialize-boost-thread-in-object-constructor
  boost::thread t(&LocalDataLoader::ThreadFunction, this);
  thread_.swap(t);
}

LocalDataLoader::~LocalDataLoader() {
  auto keys = cache_.keys();
  for (auto &key : keys) {
    auto cache_entry = cache_.get(key);
    if (cache_entry != nullptr && cache_entry->has_filename())
      try { fs::remove(fs::path(cache_entry->filename())); } catch(...) {}
  }

  is_stopping = true;
  if (thread_.joinable()) {
    thread_.join();
  }
}

bool LocalDataLoader::AddBatch(const AddBatchArgs& args) {
  if (!args.has_batch() && !args.has_batch_file_name()) {
    std::string message = "AddBatchArgs.batch or AddBatchArgs.batch_file_name must be specified";
    BOOST_THROW_EXCEPTION(InvalidOperation(message));
  }

  int timeout = args.timeout_milliseconds();
  MasterComponentConfig config = instance()->schema()->config();

  std::shared_ptr<Batch> batch = std::make_shared< ::artm::Batch>();
  if (args.has_batch_file_name()) {
    ::artm::core::BatchHelpers::LoadMessage(args.batch_file_name(), batch.get());
    ::artm::core::BatchHelpers::PopulateClassId(batch.get());
  } else {
    batch = std::make_shared<Batch>(args.batch());  // copy constructor
    BatchHelpers::PopulateClassId(batch.get());
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

void LocalDataLoader::InvokeIteration(const InvokeIterationArgs& args) {
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

bool LocalDataLoader::WaitIdle(const WaitIdleArgs& args) {
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

void LocalDataLoader::DisposeModel(ModelName model_name) {
  auto keys = cache_.keys();
  for (auto &key : keys) {
    auto cache_entry = cache_.get(key);
    if (cache_entry == nullptr) {
      continue;
    }

    if (cache_entry->model_name() == model_name) {
      if (cache_entry->has_filename())
        try { fs::remove(fs::path(cache_entry->filename())); } catch(...) {}
      cache_.erase(key);
    }
  }
}

bool LocalDataLoader::RequestThetaMatrix(const GetThetaMatrixArgs& get_theta_args,
                                         ::artm::ThetaMatrix* theta_matrix) {
  std::string model_name = get_theta_args.model_name();
  std::vector<CacheKey> keys = cache_.keys();

  for (auto &key : keys) {
    if (key.second != model_name)
      continue;

    std::shared_ptr<DataLoaderCacheEntry> cache = cache_.get(key);
    if (cache == nullptr)
      continue;

    if (cache->has_filename()) {
      DataLoaderCacheEntry cache_reloaded;
      BatchHelpers::LoadMessage(cache->filename(), &cache_reloaded);
      BatchHelpers::PopulateThetaMatrixFromCacheEntry(cache_reloaded, get_theta_args, theta_matrix);
    } else {
      BatchHelpers::PopulateThetaMatrixFromCacheEntry(*cache, get_theta_args, theta_matrix);
    }

    if (get_theta_args.clean_cache()) {
      cache_.erase(key);
    }
  }

  return true;
}

void LocalDataLoader::Callback(ModelIncrement* model_increment) {
  instance_->batch_manager()->Callback(model_increment);

  if (instance()->schema()->config().cache_theta()) {
    for (int cache_index = 0; cache_index < model_increment->cache_size(); ++cache_index) {
      DataLoaderCacheEntry* cache = model_increment->mutable_cache(cache_index);
      std::string uuid_str = cache->batch_uuid();
      boost::uuids::uuid uuid(boost::uuids::string_generator()(uuid_str.c_str()));
      ModelName model_name = cache->model_name();
      CacheKey cache_key(uuid, model_name);
      std::shared_ptr<DataLoaderCacheEntry> cache_entry(new DataLoaderCacheEntry());
      cache_entry->Swap(cache);

      std::shared_ptr<DataLoaderCacheEntry> old_entry = cache_.get(cache_key);
      cache_.set(cache_key, cache_entry);
      if (old_entry != nullptr && old_entry->has_filename())
        try { fs::remove(fs::path(old_entry->filename())); } catch(...) {}
    }
  }
}

void LocalDataLoader::ThreadFunction() {
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

      BatchManagerTask next_task = instance_->batch_manager()->Next();
      if (next_task.uuid.is_nil()) {
        boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));
        continue;
      }

      std::shared_ptr<const Batch> batch = DiskGeneration::batch(next_task);
      if (batch == nullptr) {
        instance_->batch_manager()->Done(next_task.uuid, ModelName());
        continue;
      }

      auto pi = std::make_shared<ProcessorInput>();
      pi->mutable_batch()->CopyFrom(*batch);
      pi->set_batch_uuid(boost::lexical_cast<std::string>(next_task.uuid));

      auto keys = cache_.keys();
      for (auto &key : keys) {
        auto cache_entry = cache_.get(key);
        if (cache_entry == nullptr) {
          continue;
        }

        if (cache_entry->batch_uuid() == pi->batch_uuid()) {
          if (cache_entry->has_filename()) {
            try {
              DataLoaderCacheEntry new_entry;
              BatchHelpers::LoadMessage(cache_entry->filename(), &new_entry);
              pi->add_cached_theta()->CopyFrom(new_entry);
            } catch (...) {
              LOG(ERROR) << "Unable to reload cache for " << cache_entry->filename();
            }
          } else {
            pi->add_cached_theta()->CopyFrom(*cache_entry);
          }
        }
      }

      DataLoader::PopulateDataStreams(*batch, pi.get());
      instance()->processor_queue()->push(pi);
    }
  }
  catch(boost::thread_interrupted&) {
    LOG(WARNING) << "thread_interrupted exception in LocalDataLoader::ThreadFunction() function";
    return;
  }
  catch (std::runtime_error& ex) {
    LOG(ERROR) << ex.what();
    throw;
  } catch(...) {
    LOG(FATAL) << "Fatal exception in LocalDataLoader::ThreadFunction() function";
    throw;
  }
}

RemoteDataLoader::RemoteDataLoader(Instance* instance)
    : DataLoader(instance),
      is_stopping(false),
      thread_() {
  // Keep this at the last action in constructor.
  // http://stackoverflow.com/questions/15751618/initialize-boost-thread-in-object-constructor
  boost::thread t(&RemoteDataLoader::ThreadFunction, this);
  thread_.swap(t);
}

RemoteDataLoader::~RemoteDataLoader() {
  is_stopping = true;
  if (thread_.joinable()) {
    thread_.join();
  }
}

void RemoteDataLoader::Callback(ModelIncrement* model_increment) {
  BatchIds processed_batches;
  for (int batch_index = 0; batch_index < model_increment->batch_uuid_size(); ++batch_index) {
    processed_batches.add_batch_id(model_increment->batch_uuid(batch_index));
  }

  int timeout = instance()->schema()->config().communication_timeout();
  make_rpcz_call_no_throw([&]() {
    Void response;
    instance()->master_component_service_proxy()->ReportBatches(processed_batches, &response, timeout);
  }, "RemoteDataLoader::Callback");
}

void RemoteDataLoader::ThreadFunction() {
  try {
    Helpers::SetThreadName(-1, "DataLoader thread");
    LOG(INFO) << "DataLoader thread started";
    for (;;) {
      if (is_stopping) {
        LOG(INFO) << "DataLoader thread stopped";
        break;
      }

      MasterComponentConfig config = instance()->schema()->config();
      int processor_queue_size = instance()->processor_queue()->size();
      int max_queue_size = config.processor_queue_max_size();
      if (processor_queue_size >= max_queue_size) {
        // Sleep and check for interrupt.
        boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));
        continue;
      }

      Int request;  // desired number of batches
      BatchIds response;
      request.set_value(max_queue_size - processor_queue_size);
      int timeout = config.communication_timeout();
      bool ok = make_rpcz_call_no_throw([&]() {
        instance()->master_component_service_proxy()->RequestBatches(request, &response, timeout);
      }, "RemoteDataLoader::ThreadFunction");

      if (!ok) continue;

      if (response.batch_id_size() == 0) {
        boost::this_thread::sleep(boost::posix_time::milliseconds(kNetworkPollingFrequency));
        continue;
      }

      BatchIds failed_batches;
      for (int batch_index = 0; batch_index < response.batch_id_size(); ++batch_index) {
        std::string batch_id = response.batch_id(batch_index);
        std::string batch_file_path = response.batch_file_path(batch_index);

        auto batch = std::make_shared< ::artm::Batch>();
        ::artm::core::BatchHelpers::LoadMessage(batch_file_path, batch.get());
        ::artm::core::BatchHelpers::PopulateClassId(batch.get());

        if (batch == nullptr) {
          LOG(ERROR) << "Unable to load batch '" << batch_id << "' from " << config.disk_path();
          failed_batches.add_batch_id(batch_id);
          continue;
        }

        auto pi = std::make_shared<ProcessorInput>();
        pi->mutable_batch()->CopyFrom(*batch);
        pi->set_batch_uuid(batch_id);

        // ToDo(alfrey): implement Theta-caching in network modus operandi
        DataLoader::PopulateDataStreams(*batch, pi.get());
        instance()->processor_queue()->push(pi);
      }

      if (failed_batches.batch_id_size() > 0) {
        make_rpcz_call_no_throw([&]() {
          Void response;
          instance()->master_component_service_proxy()->ReportBatches(failed_batches, &response, timeout);
        }, "RemoteDataLoader::ThreadFunction");
      }
    }
  }
  catch(boost::thread_interrupted&) {
    LOG(WARNING) << "thread_interrupted exception in RemoteDataLoader::ThreadFunction() function";
    return;
  }
  catch(...) {
    LOG(FATAL) << "Fatal exception in RemoteDataLoader::ThreadFunction() function";
    throw;
  }
}

}  // namespace core
}  // namespace artm
