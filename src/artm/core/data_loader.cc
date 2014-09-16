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
  if (disk_path.empty()) {
    generation_.set(std::shared_ptr<Generation>(new MemoryGeneration()));
  } else {
    generation_.set(std::shared_ptr<Generation>(new DiskGeneration(disk_path)));
  }

  // Keep this at the last action in constructor.
  // http://stackoverflow.com/questions/15751618/initialize-boost-thread-in-object-constructor
  boost::thread t(&LocalDataLoader::ThreadFunction, this);
  thread_.swap(t);
}

LocalDataLoader::~LocalDataLoader() {
  is_stopping = true;
  if (thread_.joinable()) {
    thread_.join();
  }
}

void LocalDataLoader::AddBatch(const Batch& batch) {
  MasterComponentConfig config = instance()->schema()->config();
  std::shared_ptr<Generation> next_gen = generation_.get()->Clone();
  if (config.compact_batches()) {
    auto compacted_batch = std::make_shared<Batch>();
    BatchHelpers::CompactBatch(batch, compacted_batch.get());
    BatchHelpers::PopulateClassId(compacted_batch.get());
    next_gen->AddBatch(compacted_batch);
  } else {
    auto modified_batch = std::make_shared<Batch>(batch);
    BatchHelpers::PopulateClassId(modified_batch.get());
    next_gen->AddBatch(modified_batch);
  }

  generation_.set(next_gen);
}

int LocalDataLoader::GetTotalItemsCount() const {
  return generation_.get()->GetTotalItemsCount();
}

void LocalDataLoader::InvokeIteration(int iterations_count) {
  if (iterations_count <= 0) {
    LOG(WARNING) << "DataLoader::InvokeIteration() was called with argument '"
                 << iterations_count << "'. Call is ignored.";
    return;
  }

  // Reset scores
  instance()->merger()->ForceResetScores(ModelName());

  auto latest_generation = generation_.get();
  if (latest_generation->empty()) {
    LOG(WARNING) << "DataLoader::InvokeIteration() - current generation is empty, "
                 << "please populate DataLoader data with some data";
    return;
  }

  std::vector<boost::uuids::uuid> uuids = latest_generation->batch_uuids();
  for (int iter = 0; iter < iterations_count; ++iter) {
    for (auto &uuid : uuids) {
      instance_->batch_manager()->Add(uuid);
    }
  }
}

bool LocalDataLoader::WaitIdle(int timeout) {
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

  instance()->merger()->ForcePushTopicModelIncrement();
  instance()->merger()->ForcePullTopicModel();
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
      cache_.erase(key);
    }
  }
}

bool LocalDataLoader::RequestThetaMatrix(ModelName model_name, ::artm::ThetaMatrix* theta_matrix) {
  std::shared_ptr<Generation> generation = generation_.get();
  std::vector<boost::uuids::uuid> batch_uuids = generation->batch_uuids();

  theta_matrix->set_model_name(model_name);
  for (auto &batch_uuid : batch_uuids) {
    auto cache = cache_.get(CacheKey(batch_uuid, model_name));
    if (cache == nullptr) {
      LOG(INFO) << "Unable to find cache entry for model: " << model_name << ", batch: " << batch_uuid;
      continue;
    }

    for (int item_index = 0; item_index < cache->item_id_size(); ++item_index) {
      theta_matrix->add_item_id(cache->item_id(item_index));
      theta_matrix->add_item_weights()->CopyFrom(cache->theta(item_index));
    }
  }

  return true;
}

void LocalDataLoader::Callback(std::shared_ptr<const ModelIncrement> model_increment) {
  instance_->batch_manager()->Callback(model_increment);

  bool is_single_batch = (model_increment->batch_uuid_size() == 1);
  if (is_single_batch && instance()->schema()->config().cache_theta()) {
    for (int batch_index = 0; batch_index < model_increment->batch_uuid_size(); ++batch_index) {
      std::string uuid_str = model_increment->batch_uuid(batch_index);
      boost::uuids::uuid uuid(boost::uuids::string_generator()(uuid_str.c_str()));
      ModelName model_name = model_increment->model_name();
      CacheKey cache_key(uuid, model_name);
      std::shared_ptr<DataLoaderCacheEntry> cache_entry(new DataLoaderCacheEntry());
      cache_entry->set_batch_uuid(uuid_str);
      cache_entry->set_model_name(model_name);
      for (int item_index = 0; item_index < model_increment->item_id_size(); ++item_index) {
        cache_entry->add_item_id(model_increment->item_id(item_index));
        cache_entry->add_theta()->CopyFrom(model_increment->theta(item_index));
      }

      cache_.set(cache_key, cache_entry);
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

      boost::uuids::uuid next_batch_uuid = instance_->batch_manager()->Next();
      if (next_batch_uuid.is_nil()) {
        boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));
        continue;
      }

      auto latest_generation = generation_.get();
      std::shared_ptr<const Batch> batch = latest_generation->batch(next_batch_uuid);
      if (batch == nullptr) {
        instance_->batch_manager()->Done(next_batch_uuid, ModelName());
        continue;
      }

      auto pi = std::make_shared<ProcessorInput>();
      pi->mutable_batch()->CopyFrom(*batch);
      pi->set_batch_uuid(boost::lexical_cast<std::string>(next_batch_uuid));

      auto keys = cache_.keys();
      for (auto &key : keys) {
        auto cache_entry = cache_.get(key);
        if (cache_entry == nullptr) {
          continue;
        }

        if (cache_entry->batch_uuid() == pi->batch_uuid()) {
          pi->add_cached_theta()->CopyFrom(*cache_entry);
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
  catch(...) {
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

void RemoteDataLoader::Callback(std::shared_ptr<const ModelIncrement> model_increment) {
  BatchIds processed_batches;
  for (int batch_index = 0; batch_index < model_increment->batch_uuid_size(); ++batch_index) {
    processed_batches.add_batch_id(model_increment->batch_uuid(batch_index));
  }

  Void response;
  try {
    instance()->master_component_service_proxy()->ReportBatches(processed_batches, &response);
  } catch(...) {
    LOG(ERROR) << "Unable to report processed batches to master.";
  }
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
      try {
        instance()->master_component_service_proxy()->RequestBatches(request, &response);
      } catch(const std::runtime_error& exception) {
        LOG(ERROR) << exception.what();
      } catch(...) {
        LOG(ERROR) << "Unable to request batches from master.";
        return;
      }

      if (response.batch_id_size() == 0) {
        boost::this_thread::sleep(boost::posix_time::milliseconds(kNetworkPollingFrequency));
        continue;
      }

      BatchIds failed_batches;
      for (int batch_index = 0; batch_index < response.batch_id_size(); ++batch_index) {
        std::string batch_id = response.batch_id(batch_index);
        boost::uuids::uuid next_batch_uuid(boost::uuids::string_generator()(batch_id.c_str()));
        std::shared_ptr<Batch> batch =
          BatchHelpers::LoadBatch(next_batch_uuid, config.disk_path());

        if (batch == nullptr) {
          LOG(ERROR) << "Unable to load batch '" << batch_id << "' from " << config.disk_path();
          failed_batches.add_batch_id(batch_id);
          continue;
        }

        auto pi = std::make_shared<ProcessorInput>();
        pi->mutable_batch()->CopyFrom(*batch);
        pi->set_batch_uuid(boost::lexical_cast<std::string>(next_batch_uuid));

        // ToDo(alfrey): implement Theta-caching in network modus operandi
        DataLoader::PopulateDataStreams(*batch, pi.get());
        instance()->processor_queue()->push(pi);
      }

      if (failed_batches.batch_id_size() > 0) {
        Void response;
        try {
          instance()->master_component_service_proxy()->ReportBatches(failed_batches, &response);
        } catch(...) {
          LOG(ERROR) << "Unable to report failed batches to master.";
        }
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
