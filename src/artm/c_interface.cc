// Copyright 2017, Additive Regularization of Topic Models.

#include "artm/c_interface.h"

#include <string>
#include <iostream>  // NOLINT

#include "boost/filesystem.hpp"
#include "boost/thread/tss.hpp"
#include "boost/thread/thread.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"

#include "glog/logging.h"

#include "artm/score_calculator_interface.h"
#include "artm/version.h"
#include "artm/core/common.h"
#include "artm/core/check_messages.h"
#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/master_component.h"
#include "artm/core/template_manager.h"
#include "artm/core/collection_parser.h"
#include "artm/core/batch_manager.h"
#include "artm/core/protobuf_serialization.h"

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

  LOG(ERROR) << error;
  last_error_->assign(error);
}

static void EnableLogging(artm::ConfigureLoggingArgs* args) {
  static bool logging_enabled = false;

  if (logging_enabled && args != nullptr && args->has_log_dir() && (FLAGS_log_dir != args->log_dir())) {
    BOOST_THROW_EXCEPTION(::artm::core::InvalidOperation(
      "Logging directory can't be change after the logging started."));
  }
  if (!logging_enabled && args != nullptr && args->has_log_dir()) {
    if (!boost::filesystem::exists(args->log_dir()) || !boost::filesystem::is_directory(args->log_dir())) {
      BOOST_THROW_EXCEPTION(::artm::core::InvalidOperation(
        "Can not enable logging to " + args->log_dir() + ", check that the folder exist"));
    }
  }

  // Setting all other flags except log_dir
  if (args != nullptr) {
    if (args->has_minloglevel()) {
      FLAGS_minloglevel = args->minloglevel();
    }

    if (args->has_stderrthreshold()) {
      FLAGS_stderrthreshold = args->stderrthreshold();
    }

    if (args->has_logtostderr()) {
      FLAGS_logtostderr = args->logtostderr();
    }

    if (args->has_colorlogtostderr()) {
      FLAGS_colorlogtostderr = args->colorlogtostderr();
    }

    if (args->has_alsologtostderr()) {
      FLAGS_alsologtostderr = args->alsologtostderr();
    }

    if (args->has_logbufsecs()) {
      FLAGS_logbufsecs = args->logbufsecs();
    }

    if (args->has_logbuflevel()) {
      FLAGS_logbuflevel = args->logbuflevel();
    }

    if (args->has_max_log_size()) {
      FLAGS_max_log_size = args->max_log_size();
    }
    if (args->has_stop_logging_if_full_disk()) {
      FLAGS_stop_logging_if_full_disk = args->stop_logging_if_full_disk();
    }

    // ::google::SetVLOGLevel() is not supported in non-gcc compilers
    // https://groups.google.com/forum/#!topic/google-glog/f8D7qpXLWXw
    // if (args->has_v()) FLAGS_v = args->v();
  }

  // Special treatment for log_dir
  if (!logging_enabled) {
    std::string log_dir = args != nullptr && args->has_log_dir() ? args->log_dir() : ".";
    FLAGS_log_dir = log_dir;

    ::google::InitGoogleLogging("bigartm");

    logging_enabled = true;
    LOG(INFO) << "Logging enabled to " << log_dir.c_str();
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

static void ParseFromArray(const char* buffer, int64_t length, google::protobuf::Message* message) {
  ::artm::core::ProtobufSerialization::singleton().ParseFromArray(buffer, length, message);
}

static void SerializeToString(const google::protobuf::Message& message, std::string* output) {
  ::artm::core::ProtobufSerialization::singleton().SerializeToString(message, output);
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

const char* ArtmGetVersion() {
  static std::string version(
    boost::lexical_cast<std::string>(ARTM_VERSION_MAJOR)+"." +
    boost::lexical_cast<std::string>(ARTM_VERSION_MINOR)+"." +
    boost::lexical_cast<std::string>(ARTM_VERSION_PATCH));
  return version.c_str();
}

int64_t ArtmConfigureLogging(int64_t length, const char* configure_logging_args) {
  try {
    ::artm::ConfigureLoggingArgs args;
    ParseFromArray(configure_logging_args, length, &args);
    EnableLogging(&args);
    std::string description = ::artm::core::DescribeMessage(args);
    LOG_IF(INFO, !description.empty()) << "EnableLogging with " << description;
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int64_t ArtmSetProtobufMessageFormatToJson() {
  ::artm::core::ProtobufSerialization::singleton().SetFormatToJson();
  return ARTM_SUCCESS;
}

int64_t ArtmSetProtobufMessageFormatToBinary() {
  ::artm::core::ProtobufSerialization::singleton().SetFormatToBinary();
  return ARTM_SUCCESS;
}

int64_t ArtmProtobufMessageFormatIsJson() {
  return ::artm::core::ProtobufSerialization::singleton().IsJson();
}

int64_t ArtmCopyRequestImpl(int64_t length, char* address, std::string* source) {
  try {
    if (source == nullptr) {
      std::stringstream ss;
      ss << "There is no data to copy; check if ArtmRequestXxx method is executed before copying the result";
      set_last_error(ss.str().c_str());
      return ARTM_INVALID_OPERATION;
    }

    if (length != static_cast<int64_t>(source->size())) {
      std::stringstream ss;
      ss << "Invalid 'length' parameter ";
      ss << "(" << source->size() << " expected, found " << length << ").";
      set_last_error(ss.str());
      return ARTM_INVALID_OPERATION;
    }

    memcpy(address, StringAsArray(source), length);

    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int64_t ArtmCopyRequestedMessage(int64_t length, char* address) {
  LOG(INFO) << "ArtmCopyRequestedMessage is copying " << length << " bytes...";
  return ArtmCopyRequestImpl(length, address, last_message());
}

int64_t ArtmCopyRequestedObject(int64_t length, char* address) {
  LOG(INFO) << "ArtmCopyRequestedObject is copying " << length << " bytes...";
  return ArtmCopyRequestImpl(length, address, last_message_ex());
}

int64_t ArtmSaveBatch(const char* disk_path, int64_t length, const char* batch) {
  try {
    EnableLogging();
    artm::Batch batch_object;
    ParseFromArray(batch, length, &batch_object);
    artm::core::FixAndValidateMessage(&batch_object);
    artm::core::Helpers::SaveBatch(batch_object, std::string(disk_path), batch_object.id());
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int64_t ArtmDuplicateMasterComponent(int master_id, int64_t length, const char* duplicate_master_args) {
  try {
    EnableLogging();

    std::shared_ptr< ::artm::core::MasterComponent> master = master_component(master_id);
    auto& mcm = MasterComponentManager::singleton();
    int retval = mcm.Store(master->Duplicate());
    LOG(INFO) << "Copying MasterComponent (id=" << master_id << " to id=" << retval << ")...";
    return retval;
  } CATCH_EXCEPTIONS;
}

int64_t ArtmCreateMasterModel(int64_t length, const char* master_model_config) {
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

int64_t ArtmAsyncProcessBatches(int master_id, int64_t length, const char* process_batches_args) {
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

int64_t ArtmAwaitOperation(int operation_id, int64_t length, const char* await_operation_args) {
  try {
    artm::AwaitOperationArgs args;
    ParseFromArray(await_operation_args, length, &args);

    AsyncProcessBatchesManager& manager = AsyncProcessBatchesManager::singleton();
    std::shared_ptr<artm::core::BatchManager> batch_manager = manager.Get(operation_id);

    const int timeout = args.timeout_milliseconds();
    auto time_start = boost::posix_time::microsec_clock::local_time();
    for (;;) {
      if (batch_manager->IsEverythingProcessed()) {
        return ARTM_SUCCESS;
      }

      boost::this_thread::sleep(boost::posix_time::milliseconds(::artm::core::kIdleLoopFrequency));
      auto time_end = boost::posix_time::microsec_clock::local_time();
      if (timeout >= 0) {
        if ((time_end - time_start).total_milliseconds() >= timeout) {
          break;
        }
      }
    }

    set_last_error("The operation is still in progress. Call ArtmAwaitOperation() later.");
    return ARTM_STILL_WORKING;
  } CATCH_EXCEPTIONS;
}

int64_t ArtmAttachModel(int master_id, int64_t length, const char* attach_model_args,
                        int64_t address_length, char* address) {
  try {
    artm::AttachModelArgs args;
    ParseFromArray(attach_model_args, length, &args);
    master_component(master_id)->AttachModel(args, address_length, reinterpret_cast<float *>(address));
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int64_t ArtmDisposeMasterComponent(int master_id) {
  try {
    MasterComponentManager::singleton().Erase(master_id);
    LOG(INFO) << "Disposing MasterComponent (id=" << master_id << ")...";
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int64_t ArtmParseCollection(int64_t length, const char* collection_parser_config) {
  try {
    EnableLogging();
    artm::CollectionParserConfig config;
    ParseFromArray(collection_parser_config, length, &config);
    ::artm::core::ValidateMessage(config, /* throw_error =*/ true);
    ::artm::core::CollectionParser collection_parser(config);
    ::artm::CollectionParserInfo result = collection_parser.Parse();
    SerializeToString(result, last_message());
    return static_cast<int64_t>(last_message()->size());
  } CATCH_EXCEPTIONS;
}

int64_t ArtmRequestLoadBatch(const char* filename) {
  try {
    EnableLogging();
    auto batch = std::make_shared< ::artm::Batch>();
    ::artm::core::Helpers::LoadMessage(filename, batch.get());
    SerializeToString(*batch, last_message());
    return static_cast<int64_t>(last_message()->size());
  } CATCH_EXCEPTIONS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// EXECUTE routines (public ARTM interface)
///////////////////////////////////////////////////////////////////////////////////////////////////

// Execute a method of MasterComponent with explicitly provided args
template<typename FuncT>
int64_t ArtmExecute(int master_id, const char* args, FuncT func) {
  try {
    LOG(INFO) << "Pass " << args << " to " << typeid(FuncT).name();
    (master_component(master_id).get()->*func)(args);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

// Execute a method of MasterComponent with args parsed from a protobuf blob
template<typename ArgsT, typename FuncT>
int64_t ArtmExecute(int master_id, int64_t length, const char* args_blob, FuncT func) {
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
int64_t ArtmExecute(int master_id, int64_t length, const char* args_blob, const char* name, FuncT func) {
  try {
    ArgsT args;
    ParseFromArray(args_blob, length, &args);

    if (name != nullptr) {
      args.set_name(name);
    } else {
      args.clear_name();
    }

    ::artm::core::FixAndValidateMessage(&args, /* throw_error =*/ true);
    std::string description = ::artm::core::DescribeMessage(args);
    LOG_IF(INFO, !description.empty()) << "Pass " << description << " to " << typeid(FuncT).name();
    (master_component(master_id).get()->*func)(args);
    return ARTM_SUCCESS;
  } CATCH_EXCEPTIONS;
}

int64_t ArtmImportBatches(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::ImportBatchesArgs>(master_id, length, args, &MasterComponent::ImportBatches);
}

int64_t ArtmMergeModel(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::MergeModelArgs>(master_id, length, args, &MasterComponent::MergeModel);
}

int64_t ArtmRegularizeModel(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::RegularizeModelArgs>(master_id, length, args, &MasterComponent::RegularizeModel);
}

int64_t ArtmNormalizeModel(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::NormalizeModelArgs>(master_id, length, args, &MasterComponent::NormalizeModel);
}

int64_t ArtmOverwriteTopicModel(int master_id, int64_t length, const char* topic_model) {
  return ArtmExecute< ::artm::TopicModel>(master_id, length, topic_model, &MasterComponent::OverwriteTopicModel);
}

int64_t ArtmOverwriteTopicModelNamed(int master_id, int64_t length, const char* topic_model, const char* name) {
  return ArtmExecute< ::artm::TopicModel>(master_id, length, topic_model, name, &MasterComponent::OverwriteTopicModel);
}

int64_t ArtmInitializeModel(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::InitializeModelArgs>(master_id, length, args, &MasterComponent::InitializeModel);
}

int64_t ArtmExportModel(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::ExportModelArgs>(master_id, length, args, &MasterComponent::ExportModel);
}

int64_t ArtmImportModel(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::ImportModelArgs>(master_id, length, args, &MasterComponent::ImportModel);
}

int64_t ArtmExportScoreTracker(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::ExportScoreTrackerArgs>(master_id, length, args, &MasterComponent::ExportScoreTracker);
}

int64_t ArtmImportScoreTracker(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::ImportScoreTrackerArgs>(master_id, length, args, &MasterComponent::ImportScoreTracker);
}

int64_t ArtmCreateRegularizer(int master_id, int64_t length, const char* config) {
  return ArtmExecute< ::artm::RegularizerConfig>(
    master_id, length, config, &MasterComponent::CreateOrReconfigureRegularizer);
}

int64_t ArtmReconfigureRegularizer(int master_id, int64_t length, const char* config) {
  return ArtmExecute< ::artm::RegularizerConfig>(
    master_id, length, config, &MasterComponent::CreateOrReconfigureRegularizer);
}

int64_t ArtmGatherDictionary(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::GatherDictionaryArgs>(master_id, length, args, &MasterComponent::GatherDictionary);
}

int64_t ArtmFilterDictionary(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::FilterDictionaryArgs>(master_id, length, args, &MasterComponent::FilterDictionary);
}

int64_t ArtmCreateDictionary(int master_id, int64_t length, const char* data) {
  return ArtmExecute< ::artm::DictionaryData>(master_id, length, data, &MasterComponent::CreateDictionary);
}

int64_t ArtmCreateDictionaryNamed(int master_id, int64_t length, const char* data, const char* name) {
  return ArtmExecute< ::artm::DictionaryData>(master_id, length, data, name, &MasterComponent::CreateDictionary);
}

int64_t ArtmImportDictionary(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::ImportDictionaryArgs>(master_id, length, args, &MasterComponent::ImportDictionary);
}

int64_t ArtmExportDictionary(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::ExportDictionaryArgs>(master_id, length, args, &MasterComponent::ExportDictionary);
}

int64_t ArtmReconfigureMasterModel(int master_id, int64_t length, const char* config) {
  return ArtmExecute< ::artm::MasterModelConfig>(master_id, length, config, &MasterComponent::ReconfigureMasterModel);
}

int64_t ArtmReconfigureTopicName(int master_id, int64_t length, const char* config) {
  return ArtmExecute< ::artm::MasterModelConfig>(master_id, length, config, &MasterComponent::ReconfigureTopicName);
}

int64_t ArtmFitOfflineMasterModel(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::FitOfflineMasterModelArgs>(master_id, length, args, &MasterComponent::FitOffline);
}

int64_t ArtmFitOnlineMasterModel(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::FitOnlineMasterModelArgs>(master_id, length, args, &MasterComponent::FitOnline);
}

int64_t ArtmClearThetaCache(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::ClearThetaCacheArgs>(master_id, length, args, &MasterComponent::ClearThetaCache);
}

int64_t ArtmClearScoreCache(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::ClearScoreCacheArgs>(master_id, length, args, &MasterComponent::ClearScoreCache);
}

int64_t ArtmClearScoreArrayCache(int master_id, int64_t length, const char* args) {
  return ArtmExecute< ::artm::ClearScoreArrayCacheArgs>(master_id, length, args,
                                                        &MasterComponent::ClearScoreArrayCache);
}

int64_t ArtmDisposeRegularizer(int master_id, const char* name) {
  return ArtmExecute(master_id, name, &MasterComponent::DisposeRegularizer);
}

int64_t ArtmDisposeModel(int master_id, const char* name) {
  return ArtmExecute(master_id, name, &MasterComponent::DisposeModel);
}

int64_t ArtmDisposeDictionary(int master_id, const char* name) {
  return ArtmExecute(master_id, name, &MasterComponent::DisposeDictionary);
}

int64_t ArtmDisposeBatch(int master_id, const char* name) {
  return ArtmExecute(master_id, name, &MasterComponent::DisposeBatch);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// REQUEST routines (public ARTM interface)
///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ResultT>
int64_t ArtmRequest(int master_id) {
  try {
    ResultT result;
    master_component(master_id)->Request(&result);
    ::artm::core::FixAndValidateMessage(&result, /* throw_error =*/ false);
    SerializeToString(result, last_message());
    return static_cast<int64_t>(last_message()->size());
  } CATCH_EXCEPTIONS;
}

template<typename ArgsT, typename ResultT>
int64_t ArtmRequest(int master_id, int64_t length, const char* args_blob) {
  try {
    ArgsT args;
    ResultT result;
    ParseFromArray(args_blob, length, &args);
    ::artm::core::FixAndValidateMessage(&args, /* throw_error =*/ true);
    std::string description = ::artm::core::DescribeMessage(args);
    LOG_IF(INFO, !description.empty()) << "Pass " << description << " to MasterComponent::Request";
    master_component(master_id)->Request(args, &result);
    ::artm::core::FixAndValidateMessage(&result, /* throw_error =*/ false);
    SerializeToString(result, last_message());
    return static_cast<int64_t>(last_message()->size());
  } CATCH_EXCEPTIONS;
}

template<typename ArgsT, typename ResultT>
int64_t ArtmRequestExternal(int master_id, int64_t length, const char* args_blob) {
  try {
    ArgsT args;
    ResultT result;
    ParseFromArray(args_blob, length, &args);
    ::artm::core::FixAndValidateMessage(&args, /* throw_error =*/ true);
    std::string description = ::artm::core::DescribeMessage(args);
    LOG_IF(INFO, !description.empty()) << "Pass " << description << " to MasterComponent::Request (extended)";
    master_component(master_id)->Request(args, &result, last_message_ex());
    ::artm::core::FixAndValidateMessage(&result, /* throw_error =*/ false);
    SerializeToString(result, last_message());
    return static_cast<int64_t>(last_message()->size());
  } CATCH_EXCEPTIONS;
}

int64_t ArtmRequestScore(int master_id, int64_t length, const char* args) {
  return ArtmRequest< ::artm::GetScoreValueArgs,
                      ::artm::ScoreData>(master_id, length, args);
}

int64_t ArtmRequestScoreArray(int master_id, int64_t length, const char* args) {
  return ArtmRequest< ::artm::GetScoreArrayArgs,
                      ::artm::ScoreArray>(master_id, length, args);
}

int64_t ArtmRequestDictionary(int master_id, int64_t length, const char* args) {
  return ArtmRequest< ::artm::GetDictionaryArgs,
                      ::artm::DictionaryData>(master_id, length, args);
}

int64_t ArtmRequestMasterComponentInfo(int master_id, int64_t length, const char* args) {
  return ArtmRequest< ::artm::GetMasterComponentInfoArgs,
                      ::artm::MasterComponentInfo>(master_id, length, args);
}

int64_t ArtmRequestProcessBatches(int master_id, int64_t length, const char* args) {
  return ArtmRequest< ::artm::ProcessBatchesArgs,
                      ::artm::ProcessBatchesResult>(master_id, length, args);
}

int64_t ArtmRequestProcessBatchesExternal(int master_id, int64_t length, const char* args) {
  return ArtmRequestExternal< ::artm::ProcessBatchesArgs,
                              ::artm::ProcessBatchesResult>(master_id, length, args);
}

int64_t ArtmRequestMasterModelConfig(int master_id) {
  return ArtmRequest< ::artm::MasterModelConfig>(master_id);
}

int64_t ArtmRequestThetaMatrix(int master_id, int64_t length, const char* args) {
  return ArtmRequest< ::artm::GetThetaMatrixArgs,
                      ::artm::ThetaMatrix>(master_id, length, args);
}

int64_t ArtmRequestThetaMatrixExternal(int master_id, int64_t length, const char* args) {
  return ArtmRequestExternal< ::artm::GetThetaMatrixArgs,
                              ::artm::ThetaMatrix>(master_id, length, args);
}

int64_t ArtmRequestTopicModel(int master_id, int64_t length, const char* args) {
  return ArtmRequest< ::artm::GetTopicModelArgs,
                      ::artm::TopicModel>(master_id, length, args);
}

int64_t ArtmRequestTopicModelExternal(int master_id, int64_t length, const char* args) {
  return ArtmRequestExternal< ::artm::GetTopicModelArgs,
                              ::artm::TopicModel>(master_id, length, args);
}

int64_t ArtmRequestTransformMasterModel(int master_id, int64_t length, const char* args) {
  return ArtmRequest< ::artm::TransformMasterModelArgs,
                      ::artm::ThetaMatrix>(master_id, length, args);
}

int64_t ArtmRequestTransformMasterModelExternal(int master_id, int64_t length, const char* args) {
  return ArtmRequestExternal< ::artm::TransformMasterModelArgs,
                              ::artm::ThetaMatrix>(master_id, length, args);
}
