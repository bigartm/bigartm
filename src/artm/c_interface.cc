// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/c_interface.h"

#include <string>
#include <iostream>  // NOLINT

#include "boost/thread/tss.hpp"
#include "boost/thread/thread.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"

#include "glog/logging.h"

#include "artm/messages.pb.h"
#include "artm/score_calculator_interface.h"
#include "artm/core/common.h"
#include "artm/core/check_messages.h"
#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/master_component.h"
#include "artm/core/template_manager.h"
#include "artm/core/collection_parser.h"
#include "artm/core/batch_manager.h"

typedef artm::core::TemplateManager<std::shared_ptr< ::artm::core::MasterComponent>> MasterComponentManager;
typedef artm::core::TemplateManager<std::shared_ptr< ::artm::core::BatchManager>> AsyncProcessBatchesManager;

using ::artm::core::MasterComponent;

// Never use the following variables explicitly (only through the corresponding methods).
// It might be good idea to make them a private members of a new singleton class.
static boost::thread_specific_ptr<std::string> last_message_;
static boost::thread_specific_ptr<std::string> last_message_ex_;
static boost::thread_specific_ptr<std::string> last_error_;

static std::string* last_message() {
  if (last_message_.get() == nullptr) {
    last_message_.reset(new std::string());
  }

  return last_message_.get();
}

static std::string* last_message_ex() {
  if (last_message_ex_.get() == nullptr) {
    last_message_ex_.reset(new std::string());
  }

  return last_message_ex_.get();
}

static void set_last_error(const std::string& error) {
  if (last_error_.get() == nullptr) {
    last_error_.reset(new std::string());
  }

  last_error_->assign(error);
}

static void EnableLogging(artm::ConfigureLoggingArgs* args) {
  static bool logging_enabled = false;

  if (logging_enabled && args != nullptr && args->has_log_dir())
    LOG(WARNING) << "Logging directory can't be change after the logging started.";

  // Special treatment for log_dir
  if (!logging_enabled) {
    std::string log_dir = args != nullptr && args->has_log_dir() ? args->log_dir() : ".";
    FLAGS_log_dir = log_dir;

    ::google::InitGoogleLogging(log_dir.c_str());

    logging_enabled = true;
    LOG(INFO) << "Logging enabled to " << log_dir.c_str();
  }

  // Setting all other flags except log_dir
  if (args != nullptr) {
    if (args->has_minloglevel()) FLAGS_minloglevel = args->minloglevel();
    if (args->has_stderrthreshold()) FLAGS_stderrthreshold = args->stderrthreshold();
    if (args->has_logtostderr()) FLAGS_logtostderr = args->logtostderr();

    if (args->has_colorlogtostderr()) FLAGS_colorlogtostderr = args->colorlogtostderr();
    if (args->has_alsologtostderr()) FLAGS_alsologtostderr = args->alsologtostderr();

    if (args->has_logbufsecs()) FLAGS_logbufsecs = args->logbufsecs();
    if (args->has_logbuflevel()) FLAGS_logbuflevel = args->logbuflevel();

    if (args->has_max_log_size()) FLAGS_max_log_size = args->max_log_size();
    if (args->has_stop_logging_if_full_disk()) FLAGS_stop_logging_if_full_disk = args->stop_logging_if_full_disk();

    // ::google::SetVLOGLevel() is not supported in non-gcc compilers
    // https://groups.google.com/forum/#!topic/google-glog/f8D7qpXLWXw
    // if (args->has_v()) FLAGS_v = args->v();
  }
}

static void EnableLogging() {
  try {
    EnableLogging(nullptr);
  }
  catch (...) {
    std::cerr << "InitGoogleLogging() or glog flags modification failed.\n";
  }
}

static std::shared_ptr< ::artm::core::MasterComponent> master_component(int master_id) {
  auto master_component = MasterComponentManager::singleton().Get(master_id);
  if (master_component == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InvalidMasterIdException(boost::lexical_cast<std::string>(master_id)));
  }

  return master_component;
}

static char* StringAsArray(std::string* str) {
  return str->empty() ? NULL : &*str->begin();
}

static void ParseFromArray(const char* buffer, int length, google::protobuf::Message* message) {
  if (!message->ParseFromArray(buffer, length))
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException("Unable to parse the message"));
}

// =========================================================================
// Misc routines (public ARTM interface)
// =========================================================================

const char* ArtmGetLastErrorMessage() {
  if (last_error_.get() == nullptr) {
    return nullptr;
  }

  return last_error_->c_str();
}

int ArtmConfigureLogging(int length, const char* configure_logging_args) {
  try {
    ::artm::ConfigureLoggingArgs args;
    ParseFromArray(configure_logging_args, length, &args);
    EnableLogging(&args);
    std::string description = ::artm::core::DescribeMessage(args);
    LOG_IF(INFO, !description.empty()) << "EnableLogging with " << description;
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmUpgradeBatch_v07(const char* source_path, const char* target_path) {
  try {
    ::artm::core::BatchHelpers::UpgradeBatch_v07(source_path, target_path);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmCopyRequestResult(int length, char* address) {
  ::artm::CopyRequestResultArgs args;
  std::string blob = args.SerializeAsString();
  return ArtmCopyRequestResultEx(length, address, blob.size(), blob.c_str());
}

int ArtmCopyRequestResultEx(int length, char* address, int args_length, const char* copy_result_args) {
  try {
    ::artm::CopyRequestResultArgs args;
    ParseFromArray(copy_result_args, args_length, &args);

    std::string* source = nullptr;
    if (args.request_type() == artm::CopyRequestResultArgs_RequestType_DefaultRequestType) source = last_message();
    if (args.request_type() == artm::CopyRequestResultArgs_RequestType_GetThetaSecondPass) source = last_message_ex();
    if (args.request_type() == artm::CopyRequestResultArgs_RequestType_GetModelSecondPass) source = last_message_ex();

    if (source == nullptr) {
      std::stringstream ss;
      ss << "CopyRequestResultArgs.request_type=" << args.request_type()  << " is not valid.";
      set_last_error(ss.str().c_str());
      return ARTM_INVALID_OPERATION;
    }

    if (length != static_cast<int>(source->size())) {
      std::stringstream ss;
      ss << "ArtmCopyRequestResultEx() called with invalid 'length' parameter ";
      ss << "(" << source->size() << " expected, found " << length << ").";
      set_last_error(ss.str());
      return ARTM_INVALID_OPERATION;
    }

    memcpy(address, StringAsArray(source), length);
    LOG(INFO) << "ArtmCopyRequestResultEx(request_type=" << args.request_type() << ") copied " << length << " bytes";

    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmSaveBatch(const char* disk_path, int length, const char* batch) {
  try {
    EnableLogging();
    artm::Batch batch_object;
    ParseFromArray(batch, length, &batch_object);
    artm::core::FixAndValidateMessage(&batch_object);
    artm::core::BatchHelpers::SaveBatch(batch_object, std::string(disk_path), batch_object.id());
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmDuplicateMasterComponent(int master_id, int length, const char* duplicate_master_args) {
  try {
    EnableLogging();

    std::shared_ptr< ::artm::core::MasterComponent> master = master_component(master_id);
    auto& mcm = MasterComponentManager::singleton();
    int retval = mcm.Store(master->Duplicate());
    LOG(INFO) << "Copying MasterComponent (id=" << master_id << " to id=" << retval << ")...";
    return retval;
  } CATCH_EXCEPTIONS;
}

int ArtmCreateMasterModel(int length, const char* master_model_config) {
  try {
    EnableLogging();

    artm::MasterModelConfig config;
    ParseFromArray(master_model_config, length, &config);
    ::artm::core::FixAndValidateMessage(&config, /* throw_error =*/ true);
    auto& mcm = MasterComponentManager::singleton();
    int retval = mcm.Store(std::make_shared< ::artm::core::MasterComponent>(config));
    LOG(INFO) << "Creating MasterModel (id=" << retval << ")...";
    return retval;
  } CATCH_EXCEPTIONS;
}

int ArtmAsyncProcessBatches(int master_id, int length, const char* process_batches_args) {
  try {
    artm::ProcessBatchesArgs args;
    ParseFromArray(process_batches_args, length, &args);
    ::artm::core::FixAndValidateMessage(&args, /* throw_error =*/ true);
    std::string description = ::artm::core::DescribeMessage(args);
    LOG_IF(INFO, !description.empty()) << "Pass " << description << " to MasterComponent::AsyncRequestProcessBatches";
    std::shared_ptr< ::artm::core::MasterComponent> master = master_component(master_id);

    std::shared_ptr< ::artm::core::BatchManager> batch_manager = std::make_shared< ::artm::core::BatchManager>();
    master->AsyncRequestProcessBatches(args, batch_manager.get());
    int retval = AsyncProcessBatchesManager::singleton().Store(batch_manager);

    LOG(INFO) << "Creating async operation (id=" << retval << ")...";
    return retval;
  } CATCH_EXCEPTIONS;
}

int ArtmAwaitOperation(int operation_id, int length, const char* await_operation_args) {
  try {
    artm::AwaitOperationArgs args;
    ParseFromArray(await_operation_args, length, &args);

    AsyncProcessBatchesManager& manager = AsyncProcessBatchesManager::singleton();
    std::shared_ptr<artm::core::BatchManager> batch_manager = manager.Get(operation_id);

    const int timeout = args.timeout_milliseconds();
    auto time_start = boost::posix_time::microsec_clock::local_time();
    for (;;) {
      if (batch_manager->IsEverythingProcessed())
        return ARTM_SUCCESS;

      boost::this_thread::sleep(boost::posix_time::milliseconds(::artm::core::kIdleLoopFrequency));
      auto time_end = boost::posix_time::microsec_clock::local_time();
      if (timeout >= 0) {
        if ((time_end - time_start).total_milliseconds() >= timeout)
          break;
      }
    }

    set_last_error("The operation is still in progress. Call ArtmAwaitOperation() later.");
    return ARTM_STILL_WORKING;
  } CATCH_EXCEPTIONS;
}

int ArtmAttachModel(int master_id, int length, const char* attach_model_args, int address_length, char* address) {
  try {
    artm::AttachModelArgs args;
    ParseFromArray(attach_model_args, length, &args);
    master_component(master_id)->AttachModel(args, address_length, reinterpret_cast<float *>(address));
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmDisposeMasterComponent(int master_id) {
  try {
    MasterComponentManager::singleton().Erase(master_id);
    LOG(INFO) << "Disposing MasterComponent (id=" << master_id << ")...";
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmParseCollection(int length, const char* collection_parser_config) {
  try {
    EnableLogging();
    artm::CollectionParserConfig config;
    ParseFromArray(collection_parser_config, length, &config);
    ::artm::core::ValidateMessage(config, /* throw_error =*/ true);
    ::artm::core::CollectionParser collection_parser(config);
    collection_parser.Parse();
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmRequestLoadBatch(const char* filename) {
  try {
    EnableLogging();
    auto batch = std::make_shared< ::artm::Batch>();
    ::artm::core::BatchHelpers::LoadMessage(filename, batch.get());
    batch->SerializeToString(last_message());
    return last_message()->size();
  } CATCH_EXCEPTIONS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// EXECUTE routines (public ARTM interface)
///////////////////////////////////////////////////////////////////////////////////////////////////

// Execute a method of MasterComponent with explicitly provided args
template<typename FuncT>
int ArtmExecute(int master_id, const char* args, FuncT func) {
  try {
    LOG(INFO) << "Pass " << args << " to " << typeid(FuncT).name();
    (master_component(master_id).get()->*func)(args);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

// Execute a method of MasterComponent with args parsed from a protobuf blob
template<typename ArgsT, typename FuncT>
int ArtmExecute(int master_id, int length, const char* args_blob, FuncT func) {
  try {
    ArgsT args;
    ParseFromArray(args_blob, length, &args);
    ::artm::core::FixAndValidateMessage(&args, /* throw_error =*/ true);
    std::string description = ::artm::core::DescribeMessage(args);
    LOG_IF(INFO, !description.empty()) << "Pass " << description << " to " << typeid(FuncT).name();
    (master_component(master_id).get()->*func)(args);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

// Execute a method of MasterComponent with args parsed from a protobuf blob (name is overwritten)
template<typename ArgsT, typename FuncT>
int ArtmExecute(int master_id, int length, const char* args_blob, const char* name, FuncT func) {
  try {
    ArgsT args;
    ParseFromArray(args_blob, length, &args);

    if (name != nullptr)
      args.set_name(name);
    else
      args.clear_name();

    ::artm::core::FixAndValidateMessage(&args, /* throw_error =*/ true);
    std::string description = ::artm::core::DescribeMessage(args);
    LOG_IF(INFO, !description.empty()) << "Pass " << description << " to " << typeid(FuncT).name();
    (master_component(master_id).get()->*func)(args);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int ArtmImportBatches(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::ImportBatchesArgs>(master_id, length, args, &MasterComponent::ImportBatches);
}

int ArtmMergeModel(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::MergeModelArgs>(master_id, length, args, &MasterComponent::MergeModel);
}

int ArtmRegularizeModel(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::RegularizeModelArgs>(master_id, length, args, &MasterComponent::RegularizeModel);
}

int ArtmNormalizeModel(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::NormalizeModelArgs>(master_id, length, args, &MasterComponent::NormalizeModel);
}

int ArtmOverwriteTopicModel(int master_id, int length, const char* topic_model) {
  return ArtmExecute< ::artm::TopicModel>(master_id, length, topic_model, &MasterComponent::OverwriteTopicModel);
}

int ArtmOverwriteTopicModelNamed(int master_id, int length, const char* topic_model, const char* name) {
  return ArtmExecute< ::artm::TopicModel>(master_id, length, topic_model, name, &MasterComponent::OverwriteTopicModel);
}

int ArtmInitializeModel(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::InitializeModelArgs>(master_id, length, args, &MasterComponent::InitializeModel);
}

int ArtmExportModel(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::ExportModelArgs>(master_id, length, args, &MasterComponent::ExportModel);
}

int ArtmImportModel(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::ImportModelArgs>(master_id, length, args, &MasterComponent::ImportModel);
}

int ArtmCreateRegularizer(int master_id, int length, const char* config) {
  return ArtmExecute< ::artm::RegularizerConfig>(
    master_id, length, config, &MasterComponent::CreateOrReconfigureRegularizer);
}

int ArtmReconfigureRegularizer(int master_id, int length, const char* config) {
  return ArtmExecute< ::artm::RegularizerConfig>(
    master_id, length, config, &MasterComponent::CreateOrReconfigureRegularizer);
}

int ArtmGatherDictionary(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::GatherDictionaryArgs>(master_id, length, args, &MasterComponent::GatherDictionary);
}

int ArtmFilterDictionary(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::FilterDictionaryArgs>(master_id, length, args, &MasterComponent::FilterDictionary);
}

int ArtmCreateDictionary(int master_id, int length, const char* data) {
  return ArtmExecute< ::artm::DictionaryData>(master_id, length, data, &MasterComponent::CreateDictionary);
}

int ArtmCreateDictionaryNamed(int master_id, int length, const char* data, const char* name) {
  return ArtmExecute< ::artm::DictionaryData>(master_id, length, data, name, &MasterComponent::CreateDictionary);
}

int ArtmImportDictionary(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::ImportDictionaryArgs>(master_id, length, args, &MasterComponent::ImportDictionary);
}

int ArtmExportDictionary(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::ExportDictionaryArgs>(master_id, length, args, &MasterComponent::ExportDictionary);
}

int ArtmReconfigureMasterModel(int master_id, int length, const char* config) {
  return ArtmExecute< ::artm::MasterModelConfig>(master_id, length, config, &MasterComponent::ReconfigureMasterModel);
}

int ArtmFitOfflineMasterModel(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::FitOfflineMasterModelArgs>(master_id, length, args, &MasterComponent::FitOffline);
}

int ArtmFitOnlineMasterModel(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::FitOnlineMasterModelArgs>(master_id, length, args, &MasterComponent::FitOnline);
}

int ArtmClearThetaCache(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::ClearThetaCacheArgs>(master_id, length, args, &MasterComponent::ClearThetaCache);
}

int ArtmClearScoreCache(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::ClearScoreCacheArgs>(master_id, length, args, &MasterComponent::ClearScoreCache);
}

int ArtmClearScoreArrayCache(int master_id, int length, const char* args) {
  return ArtmExecute< ::artm::ClearScoreArrayCacheArgs>(master_id, length, args,
                                                        &MasterComponent::ClearScoreArrayCache);
}

int ArtmDisposeRegularizer(int master_id, const char* name) {
  return ArtmExecute(master_id, name, &MasterComponent::DisposeRegularizer);
}

int ArtmDisposeModel(int master_id, const char* name) {
  return ArtmExecute(master_id, name, &MasterComponent::DisposeModel);
}

int ArtmDisposeDictionary(int master_id, const char* name) {
  return ArtmExecute(master_id, name, &MasterComponent::DisposeDictionary);
}

int ArtmDisposeBatch(int master_id, const char* name) {
  return ArtmExecute(master_id, name, &MasterComponent::DisposeBatch);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// REQUEST routines (public ARTM interface)
///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ResultT>
int ArtmRequest(int master_id) {
  try {
    ResultT result;
    master_component(master_id)->Request(&result);
    ::artm::core::ValidateMessage(result, /* throw_error =*/ false);
    result.SerializeToString(last_message());
    return last_message()->size();
  } CATCH_EXCEPTIONS;
}

template<typename ArgsT, typename ResultT>
int ArtmRequest(int master_id, int length, const char* args_blob) {
  try {
    ArgsT args;
    ResultT result;
    ParseFromArray(args_blob, length, &args);
    ::artm::core::FixAndValidateMessage(&args, /* throw_error =*/ true);
    std::string description = ::artm::core::DescribeMessage(args);
    LOG_IF(INFO, !description.empty()) << "Pass " << description << " to MasterComponent::Request";
    master_component(master_id)->Request(args, &result);
    ::artm::core::ValidateMessage(result, /* throw_error =*/ false);
    result.SerializeToString(last_message());
    return last_message()->size();
  } CATCH_EXCEPTIONS;
}

template<typename ArgsT, typename ResultT>
int ArtmRequestExternal(int master_id, int length, const char* args_blob) {
  try {
    ArgsT args;
    ResultT result;
    ParseFromArray(args_blob, length, &args);
    ::artm::core::FixAndValidateMessage(&args, /* throw_error =*/ true);
    std::string description = ::artm::core::DescribeMessage(args);
    LOG_IF(INFO, !description.empty()) << "Pass " << description << " to MasterComponent::Request (extended)";
    master_component(master_id)->Request(args, &result, last_message_ex());
    ::artm::core::ValidateMessage(result, /* throw_error =*/ false);
    result.SerializeToString(last_message());
    return last_message()->size();
  } CATCH_EXCEPTIONS;
}

int ArtmRequestScore(int master_id, int length, const char* args) {
  return ArtmRequest< ::artm::GetScoreValueArgs,
                      ::artm::ScoreData>(master_id, length, args);
}

int ArtmRequestScoreArray(int master_id, int length, const char* args) {
  return ArtmRequest< ::artm::GetScoreArrayArgs,
                      ::artm::ScoreDataArray>(master_id, length, args);
}

int ArtmRequestDictionary(int master_id, int length, const char* args) {
  return ArtmRequest< ::artm::GetDictionaryArgs,
                      ::artm::DictionaryData>(master_id, length, args);
}

int ArtmRequestMasterComponentInfo(int master_id, int length, const char* args) {
  return ArtmRequest< ::artm::GetMasterComponentInfoArgs,
                      ::artm::MasterComponentInfo>(master_id, length, args);
}

int ArtmRequestProcessBatches(int master_id, int length, const char* args) {
  return ArtmRequest< ::artm::ProcessBatchesArgs,
                      ::artm::ProcessBatchesResult>(master_id, length, args);
}

int ArtmRequestProcessBatchesExternal(int master_id, int length, const char* args) {
  return ArtmRequestExternal< ::artm::ProcessBatchesArgs,
                              ::artm::ProcessBatchesResult>(master_id, length, args);
}

int ArtmRequestMasterModelConfig(int master_id) {
  return ArtmRequest< ::artm::MasterModelConfig>(master_id);
}

int ArtmRequestThetaMatrix(int master_id, int length, const char* args) {
  return ArtmRequest< ::artm::GetThetaMatrixArgs,
                      ::artm::ThetaMatrix>(master_id, length, args);
}

int ArtmRequestThetaMatrixExternal(int master_id, int length, const char* args) {
  return ArtmRequestExternal< ::artm::GetThetaMatrixArgs,
                              ::artm::ThetaMatrix>(master_id, length, args);
}

int ArtmRequestTopicModel(int master_id, int length, const char* args) {
  return ArtmRequest< ::artm::GetTopicModelArgs,
                      ::artm::TopicModel>(master_id, length, args);
}

int ArtmRequestTopicModelExternal(int master_id, int length, const char* args) {
  return ArtmRequestExternal< ::artm::GetTopicModelArgs,
                              ::artm::TopicModel>(master_id, length, args);
}

int ArtmRequestTransformMasterModel(int master_id, int length, const char* args) {
  return ArtmRequest< ::artm::TransformMasterModelArgs,
                      ::artm::ThetaMatrix>(master_id, length, args);
}

int ArtmRequestTransformMasterModelExternal(int master_id, int length, const char* args) {
  return ArtmRequestExternal< ::artm::TransformMasterModelArgs,
                              ::artm::ThetaMatrix>(master_id, length, args);
}
