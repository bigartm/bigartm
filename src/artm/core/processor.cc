// Copyright 2017, Additive Regularization of Topic Models.

#include "artm/core/processor.h"

#include <stdlib.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>

#include "boost/exception/diagnostic_information.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_generators.hpp"
#include "boost/uuid/uuid_io.hpp"

#include "glog/logging.h"

#include "artm/core/call_on_destruction.h"
#include "artm/core/cuckoo_watch.h"
#include "artm/core/batch_manager.h"
#include "artm/core/cache_manager.h"
#include "artm/utility/blas.h"

#include "artm/core/processor_helpers.h"
#include "artm/core/processor_transaction_helpers.h"

namespace fs = boost::filesystem;

namespace artm {
namespace core {

Processor::Processor(Instance* instance)
    : instance_(instance),
      is_stopping(false),
      thread_() {
  // Keep this at the last action in constructor.
  // http://stackoverflow.com/questions/15751618/initialize-boost-thread-in-object-constructor
  boost::thread t(&Processor::ThreadFunction, this);
  thread_.swap(t);
}

Processor::~Processor() {
  is_stopping = true;
  if (thread_.joinable()) {
    thread_.join();
  }
}

void Processor::ThreadFunction() {
  try {
    int total_processed_batches = 0;  // counter

    // Do not log performance measurements below kTimeLoggingThreshold milliseconds
    const int kTimeLoggingThreshold = 0;

    Helpers::SetThreadName(-1, "Processor thread");
    LOG(INFO) << "Processor thread started";
    int pop_retries = 0;
    const int pop_retries_max = 20;

    util::Blas* blas = util::Blas::builtin();

    for (;;) {
      if (is_stopping) {
        LOG(INFO) << "Processor thread stopped";
        LOG(INFO) << "Total number of processed batches: " << total_processed_batches;
        break;
      }

      std::shared_ptr<ProcessorInput> part;
      if (!instance_->processor_queue()->try_pop(&part)) {
        pop_retries++;
        LOG_IF(INFO, pop_retries == pop_retries_max) << "No data in processing queue, waiting...";

        boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));

        continue;
      }

      LOG_IF(INFO, pop_retries >= pop_retries_max) << "Processing queue has data, processing started";
      pop_retries = 0;

      // CuckooWatch logs time from now to destruction
      const std::string batch_name = part->has_batch_filename() ? part->batch_filename() : part->batch().id();
      CuckooWatch cuckoo(std::string("ProcessBatch(") + batch_name + std::string(")"));
      total_processed_batches++;

      call_on_destruction c([&]() {  // NOLINT
        if (part->batch_manager() != nullptr) {
          part->batch_manager()->Callback(part->task_id());
        }
      });

      Batch batch;
      {
        CuckooWatch cuckoo2("LoadMessage", &cuckoo, kTimeLoggingThreshold);
        if (part->has_batch_filename()) {
          auto mem_batch = instance_->batches()->get(part->batch_filename());
          if (mem_batch != nullptr) {
            batch.CopyFrom(*mem_batch);
          } else {
            try {
              ::artm::core::Helpers::LoadMessage(part->batch_filename(), &batch);
            } catch (std::exception& ex) {
              LOG(ERROR) << ex.what() << ", the batch will be skipped.";
              continue;
            }
          }
        } else {  // part->has_batch_filename()
          batch.CopyFrom(part->batch());
        }
      }

      std::shared_ptr<MasterModelConfig> master_config = instance_->config();

      const ModelName& model_name = part->model_name();
      const ProcessBatchesArgs& args = part->args();
      {
        if (args.class_id_size() != args.class_weight_size()) {
          BOOST_THROW_EXCEPTION(InternalError(
              "model.class_id_size() != model.class_weight_size()"));
        }

        if (args.transaction_typename_size() != args.transaction_weight_size()) {
          std::stringstream ss;
          ss << "model.transaction_type_size() [ " << args.transaction_typename_size()
             << " ] != model.transaction_weight_size() [ " << args.transaction_weight_size() << " ]";
          BOOST_THROW_EXCEPTION(InternalError(ss.str()));
        }

        std::shared_ptr<const PhiMatrix> phi_matrix = instance_->GetPhiMatrix(model_name);
        if (phi_matrix == nullptr) {
          LOG(ERROR) << "Model " << model_name << " does not exist.";
          continue;
        }
        const PhiMatrix& p_wt = *phi_matrix;

        if (batch.token_size() == 0) {
          continue;
        }

        std::shared_ptr<const PhiMatrix> nwt_target;
        if (part->has_nwt_target_name()) {
          nwt_target = instance_->GetPhiMatrix(part->nwt_target_name());
          if (nwt_target == nullptr) {
            LOG(ERROR) << "Model " << part->nwt_target_name() << " does not exist.";
            continue;
          }
        }

        std::stringstream model_description;
        if (part->has_nwt_target_name()) {
          model_description << part->nwt_target_name();
        } else {
          model_description << &p_wt;
        }
        VLOG(0) << "Processor: start processing batch " << batch.id() << " into model " << model_description.str();

        std::shared_ptr<ThetaMatrix> cache;
        if (part->has_reuse_theta_cache_manager()) {
          CuckooWatch cuckoo2("FindReuseThetaCacheEntry", &cuckoo, kTimeLoggingThreshold);
          cache = part->reuse_theta_cache_manager()->FindCacheEntry(batch);
        }
        std::shared_ptr<LocalThetaMatrix<float>> theta_matrix;
        {
          CuckooWatch cuckoo2("InitializeTheta", &cuckoo, kTimeLoggingThreshold);
          theta_matrix = ProcessorHelpers::InitializeTheta(p_wt.topic_size(), batch, args, cache.get());
        }

        if (p_wt.token_size() == 0) {
          LOG(INFO) << "Phi is empty, calculations for the model " + model_name +
            "would not be processed on this iteration";
          continue;
        }

        std::shared_ptr<NwtWriteAdapter> nwt_writer;
        if (nwt_target != nullptr) {
          nwt_writer = std::make_shared<NwtWriteAdapter>(const_cast<PhiMatrix*>(nwt_target.get()));
        }

        std::shared_ptr<ThetaMatrix> new_cache_entry_ptr(nullptr);
        if (part->has_cache_manager()) {
          new_cache_entry_ptr.reset(new ThetaMatrix());
        }

        std::shared_ptr<ThetaMatrix> new_ptdw_cache_entry_ptr(nullptr);
        if (part->has_ptdw_cache_manager()) {
          new_ptdw_cache_entry_ptr.reset(new ThetaMatrix());
        }

        if (new_cache_entry_ptr != nullptr) {
          new_cache_entry_ptr->mutable_topic_name()->CopyFrom(p_wt.topic_name());
        }

        if (new_ptdw_cache_entry_ptr != nullptr) {
          new_ptdw_cache_entry_ptr->mutable_topic_name()->CopyFrom(p_wt.topic_name());
        }

        {
          RegularizeThetaAgentCollection theta_agents;
          RegularizePtdwAgentCollection ptdw_agents;
          {
            CuckooWatch cuckoo2("CreateRegularizerAgents", &cuckoo, kTimeLoggingThreshold);
            ProcessorHelpers::CreateRegularizerAgents(batch, args, instance_, &theta_agents, &ptdw_agents);
          }

          // We assum here that batch is correct, e.g. it's transaction_type field
          // in case of regular model contains ALL class_ids from batch, not their subset.
          // Both parser and checker generates such batches.
          bool use_real_transactions = true;
          if (batch.transaction_typename_size() == 1 && batch.transaction_typename(0) == DefaultTransactionTypeName) {
            use_real_transactions = false;
          }

          if (use_real_transactions) {
            if (ptdw_agents.empty() && !part->has_ptdw_cache_manager()) {
              std::shared_ptr<BatchTransactionInfo> batch_info;
              {
                CuckooWatch cuckoo2("PrepareBatchInfo", &cuckoo, kTimeLoggingThreshold);
                batch_info = ProcessorTransactionHelpers::PrepareBatchInfo(
                  batch, args, p_wt);
              }

              CuckooWatch cuckoo2("InferThetaAndUpdateNwtSparseNew", &cuckoo, kTimeLoggingThreshold);
              ProcessorTransactionHelpers::TransactionInferThetaAndUpdateNwtSparse(
                                              args, batch, part->batch_weight(),
                                              batch_info, p_wt, theta_agents,
                                              theta_matrix.get(), nwt_writer.get(),
                                              blas, new_cache_entry_ptr.get());
            } else {
              LOG(ERROR) << "Current version of BigARTM doesn't support"
                << " ptdw matrix operations with with complex transactions";
            }
          } else {
            std::shared_ptr<CsrMatrix<float>> sparse_ndw;
            {
              CuckooWatch cuckoo2("InitializeSparseNdw", &cuckoo, kTimeLoggingThreshold);
              sparse_ndw = ProcessorHelpers::InitializeSparseNdw(batch, args);
            }

            if (ptdw_agents.empty() && !part->has_ptdw_cache_manager()) {
              CuckooWatch cuckoo2("InferThetaAndUpdateNwtSparse", &cuckoo, kTimeLoggingThreshold);
              ProcessorHelpers::InferThetaAndUpdateNwtSparse(args, batch, part->batch_weight(), *sparse_ndw, p_wt,
                                                             theta_agents, theta_matrix.get(), nwt_writer.get(),
                                                             blas, new_cache_entry_ptr.get());
            } else {
              CuckooWatch cuckoo2("InferPtdwAndUpdateNwtSparse", &cuckoo, kTimeLoggingThreshold);
              ProcessorHelpers::InferPtdwAndUpdateNwtSparse(args, batch, part->batch_weight(), *sparse_ndw,
                                                            p_wt, theta_agents, ptdw_agents, theta_matrix.get(),
                                                            nwt_writer.get(), blas, new_cache_entry_ptr.get(),
                                                            new_ptdw_cache_entry_ptr.get());
            }
          }
        }

        if (new_cache_entry_ptr != nullptr) {
          CuckooWatch cuckoo2("UpdateCacheEntry", &cuckoo, kTimeLoggingThreshold);
          part->cache_manager()->UpdateCacheEntry(batch.id(), *new_cache_entry_ptr);
        }

        if (new_ptdw_cache_entry_ptr != nullptr) {
          CuckooWatch cuckoo2("UpdatePtdwCacheEntry", &cuckoo, kTimeLoggingThreshold);
          part->ptdw_cache_manager()->UpdateCacheEntry(batch.id(), *new_ptdw_cache_entry_ptr);
        }

        for (int score_index = 0; score_index < master_config->score_config_size(); ++score_index) {
          const ScoreName& score_name = master_config->score_config(score_index).name();

          auto score_calc = instance_->scores_calculators()->get(score_name);
          if (score_calc == nullptr) {
            LOG(ERROR) << "Unable to find score calculator '" << score_name << "', referenced by "
              << "model " << p_wt.model_name() << ".";
            continue;
          }

          if (!score_calc->is_cumulative()) {
            continue;
          }

          CuckooWatch cuckoo2("CalculateScore(" + score_name + ")", &cuckoo, kTimeLoggingThreshold);

          auto score_value = ProcessorHelpers::CalcScores(score_calc.get(), batch, p_wt, args, *theta_matrix);
          if (score_value != nullptr) {
            instance_->score_manager()->Append(score_name, score_value->SerializeAsString());
            if (part->score_manager() != nullptr) {
              part->score_manager()->Append(score_name, score_value->SerializeAsString());
            }
          }
        }

        VLOG(0) << "Processor: complete processing batch " << batch.id() << " into model " << model_description.str();
      }
    }
  }
  catch (...) {
    LOG(FATAL) << boost::current_exception_diagnostic_information();
  }
}

}  // namespace core
}  // namespace artm
