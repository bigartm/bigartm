// Copyright 2017, Additive Regularization of Topic Models.

#ifdef INCLUDE_STDAFX_H
#include "stdafx.h"  // NOLINT
#endif

#include <iostream>  // NOLINT

#include "google/protobuf/util/json_util.h"

#include "artm/cpp_interface.h"

namespace artm {

inline char* StringAsArray(std::string* str) {
  return str->empty() ? NULL : &*str->begin();
}

inline std::string GetLastErrorMessage() {
  return std::string(ArtmGetLastErrorMessage());
}

static void ParseMessageFromString(const std::string& string, google::protobuf::Message* message) {
  if (ArtmProtobufMessageFormatIsJson()) {
    ::google::protobuf::util::JsonStringToMessage(string, message);
  } else {
    message->ParseFromString(string);
  }
}

static void SerializeMessageToString(const google::protobuf::Message& message, std::string* output) {
  if (ArtmProtobufMessageFormatIsJson()) {
    ::google::protobuf::util::MessageToJsonString(message, output);
  } else {
    output->clear();
    message.SerializeToString(output);
  }
}

int64_t HandleErrorCode(int64_t artm_error_code) {
  // All error codes are negative. Any non-negative value is a success.
  if (artm_error_code >= 0) {
    return artm_error_code;
  }

  if (artm_error_code == ARTM_STILL_WORKING) {
    return artm_error_code;
  }

  switch (artm_error_code) {
    case ARTM_INTERNAL_ERROR:
      throw InternalError(GetLastErrorMessage());
    case ARTM_ARGUMENT_OUT_OF_RANGE:
      throw ArgumentOutOfRangeException(GetLastErrorMessage());
    case ARTM_INVALID_MASTER_ID:
      throw InvalidMasterIdException(GetLastErrorMessage());
    case ARTM_CORRUPTED_MESSAGE:
      throw CorruptedMessageException(GetLastErrorMessage());
    case ARTM_INVALID_OPERATION:
      throw InvalidOperationException(GetLastErrorMessage());
    case ARTM_DISK_READ_ERROR:
      throw DiskReadException(GetLastErrorMessage());
    case ARTM_DISK_WRITE_ERROR:
      throw DiskWriteException(GetLastErrorMessage());
    default:
      throw InternalError("Unknown error code");
  }
}

template<typename ArgsT, typename FuncT>
int64_t ArtmExecute(const ArgsT& args, FuncT func) {
  std::string blob;
  SerializeMessageToString(args, &blob);
  return HandleErrorCode(func(blob.size(), StringAsArray(&blob)));
}

template<typename ArgsT, typename FuncT>
int64_t ArtmExecute(int master_id, const ArgsT& args, FuncT func) {
  std::string blob;
  SerializeMessageToString(args, &blob);
  return HandleErrorCode(func(master_id, blob.size(), StringAsArray(&blob)));
}

template<typename ResultT>
ResultT ArtmCopyResult(int64_t length) {
  std::string result_blob;
  result_blob.resize(length);
  HandleErrorCode(ArtmCopyRequestedMessage(length, StringAsArray(&result_blob)));

  ResultT result;
  ParseMessageFromString(result_blob, &result);
  return result;
}

template<typename ResultT, typename FuncT>
ResultT ArtmRequest(int master_id, FuncT func) {
  int64_t length = func(master_id);
  return ArtmCopyResult<ResultT>(length);
}

template<typename ResultT, typename ArgsT, typename FuncT>
ResultT ArtmRequest(int master_id, const ArgsT& args, FuncT func) {
  int64_t length = ArtmExecute(master_id, args, func);
  return ArtmCopyResult<ResultT>(length);
}

template<typename ResultT, typename ArgsT, typename FuncT>
std::shared_ptr<ResultT> ArtmRequestShared(int master_id, const ArgsT& args, FuncT func) {
  return std::make_shared<ResultT>(ArtmRequest<ResultT>(master_id, args, func));
}

void ArtmRequestMatrix(int no_rows, int no_cols, Matrix* matrix) {
  if (matrix == nullptr) {
    return;
  }

  matrix->resize(no_rows, no_cols);

  int64_t length = sizeof(float) * matrix->no_columns() * matrix->no_rows();
  HandleErrorCode(ArtmCopyRequestedObject(length, reinterpret_cast<char*>(matrix->get_data())));
}

CollectionParserInfo ParseCollection(const CollectionParserConfig& config) {
  int64_t length = ArtmExecute(config, ArtmParseCollection);
  return ArtmCopyResult<CollectionParserInfo>(length);
}

void ConfigureLogging(const ConfigureLoggingArgs& args) {
  ArtmExecute(args, ArtmConfigureLogging);
}

Batch LoadBatch(std::string filename) {
  int64_t length = ArtmRequestLoadBatch(filename.c_str());
  return ArtmCopyResult<Batch>(length);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// MasterModel implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

MasterModel::MasterModel(const MasterModelConfig& config) : id_(0), is_weak_ref_(false) {
  id_ = ArtmExecute(config, ArtmCreateMasterModel);
}

MasterModel::MasterModel(int id) : id_(id), is_weak_ref_(true) {
}

MasterModel::~MasterModel() {
  if (is_weak_ref_) {
    return;
  }

  ArtmDisposeMasterComponent(id_);
}

MasterModelConfig MasterModel::config() const {
  return ArtmRequest< ::artm::MasterModelConfig>(id_, ArtmRequestMasterModelConfig);
}

void MasterModel::Reconfigure(const MasterModelConfig& config) {
  ArtmExecute(id_, config, ArtmReconfigureMasterModel);
}

void MasterModel::ReconfigureTopicName(const MasterModelConfig& config) {
  ArtmExecute(id_, config, ArtmReconfigureTopicName);
}

TopicModel MasterModel::GetTopicModel() {
  GetTopicModelArgs args;
  args.set_model_name(config().pwt_name());
  return GetTopicModel(args);
}

TopicModel MasterModel::GetTopicModel(const GetTopicModelArgs& args) {
  return ArtmRequest< ::artm::TopicModel>(id_, args, ArtmRequestTopicModel);
}

TopicModel MasterModel::GetTopicModel(Matrix* matrix) {
  GetTopicModelArgs args;
  args.set_model_name(config().pwt_name());
  return GetTopicModel(args, matrix);
}

TopicModel MasterModel::GetTopicModel(const GetTopicModelArgs& args, Matrix* matrix) {
  auto retval = ArtmRequest< ::artm::TopicModel>(id_, args, ArtmRequestTopicModelExternal);
  ArtmRequestMatrix(retval.token_size(), retval.num_topics(), matrix);
  return retval;
}

ThetaMatrix MasterModel::GetThetaMatrix() {
  GetThetaMatrixArgs args;
  return GetThetaMatrix(args);
}

ThetaMatrix MasterModel::GetThetaMatrix(const GetThetaMatrixArgs& args) {
  return ArtmRequest< ::artm::ThetaMatrix>(id_, args, ArtmRequestThetaMatrix);
}

ThetaMatrix MasterModel::GetThetaMatrix(Matrix* matrix) {
  GetThetaMatrixArgs args;
  return GetThetaMatrix(args, matrix);
}

ThetaMatrix MasterModel::GetThetaMatrix(const GetThetaMatrixArgs& args, Matrix* matrix) {
  auto retval = ArtmRequest< ::artm::ThetaMatrix>(id_, args, ArtmRequestThetaMatrixExternal);
  ArtmRequestMatrix(retval.item_id_size(), retval.num_topics(), matrix);
  return retval;
}


ThetaMatrix MasterModel::Transform(const TransformMasterModelArgs& args) {
  return ArtmRequest< ::artm::ThetaMatrix>(id_, args, ArtmRequestTransformMasterModel);
}

ThetaMatrix MasterModel::Transform(const TransformMasterModelArgs& args, Matrix* matrix) {
  auto retval = ArtmRequest< ::artm::ThetaMatrix>(id_, args, ArtmRequestTransformMasterModelExternal);
  ArtmRequestMatrix(retval.item_id_size(), retval.num_topics(), matrix);
  return retval;
}

ScoreData MasterModel::GetScore(const GetScoreValueArgs& args) {
  return ArtmRequest<ScoreData>(id_, args, ArtmRequestScore);
}

ScoreArray MasterModel::GetScoreArray(const GetScoreArrayArgs& args) {
  return ArtmRequest<ScoreArray>(id_, args, ArtmRequestScoreArray);
}

MasterComponentInfo MasterModel::info() const {
  GetMasterComponentInfoArgs args;
  return ArtmRequest<MasterComponentInfo>(id_, args, ArtmRequestMasterComponentInfo);
}

void MasterModel::ExportModel(const ExportModelArgs& args) {
  ArtmExecute(id_, args, ArtmExportModel);
}

void MasterModel::ImportModel(const ImportModelArgs& args) {
  ArtmExecute(id_, args, ArtmImportModel);
}

void MasterModel::ExportScoreTracker(const ExportScoreTrackerArgs& args) {
  ArtmExecute(id_, args, ArtmExportScoreTracker);
}

void MasterModel::ImportScoreTracker(const ImportScoreTrackerArgs& args) {
  ArtmExecute(id_, args, ArtmImportScoreTracker);
}

void MasterModel::CreateDictionary(const DictionaryData& args) {
  ArtmExecute(id_, args, ArtmCreateDictionary);
}

void MasterModel::DisposeDictionary(const std::string& dictionary_name) {
  HandleErrorCode(ArtmDisposeDictionary(id_, dictionary_name.c_str()));
}

void MasterModel::DisposeModel(const std::string& model_name) {
  HandleErrorCode(ArtmDisposeModel(id_, model_name.c_str()));
}

void MasterModel::ImportDictionary(const ImportDictionaryArgs& args) {
  ArtmExecute(id_, args, ArtmImportDictionary);
}

void MasterModel::ExportDictionary(const ExportDictionaryArgs& args) {
  ArtmExecute(id_, args, ArtmExportDictionary);
}

void MasterModel::GatherDictionary(const GatherDictionaryArgs& args) {
  ArtmExecute(id_, args, ArtmGatherDictionary);
}

void MasterModel::FilterDictionary(const FilterDictionaryArgs& args) {
  ArtmExecute(id_, args, ArtmFilterDictionary);
}

DictionaryData MasterModel::GetDictionary(const GetDictionaryArgs& args) {
  return ArtmRequest<DictionaryData>(id_, args, ArtmRequestDictionary);
}

void MasterModel::InitializeModel(const InitializeModelArgs& args) {
  ArtmExecute(id_, args, ArtmInitializeModel);
}

void MasterModel::OverwriteModel(const TopicModel& args) {
  ArtmExecute(id_, args, ArtmOverwriteTopicModel);
}

void MasterModel::ImportBatches(const ImportBatchesArgs& args) {
  ArtmExecute(id_, args, ArtmImportBatches);
}

void MasterModel::FitOnlineModel(const FitOnlineMasterModelArgs& args) {
  ArtmExecute(id_, args, ArtmFitOnlineMasterModel);
}

void MasterModel::FitOfflineModel(const FitOfflineMasterModelArgs& args) {
  ArtmExecute(id_, args, ArtmFitOfflineMasterModel);
}

void MasterModel::MergeModel(const MergeModelArgs& args) {
  ArtmExecute(id_, args, ArtmMergeModel);
}

void MasterModel::DisposeBatch(const std::string& batch_name) {
  ArtmDisposeBatch(id_, batch_name.c_str());
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

Matrix::Matrix() : no_rows_(0), no_columns_(0), data_() {
}

Matrix::Matrix(int no_rows, int no_columns) : no_rows_(no_rows), no_columns_(no_columns), data_() {
  if (no_rows <= 0 || no_columns <= 0) {
    throw ArgumentOutOfRangeException("no_rows and no_columns must be positive");
  }

  data_.resize(static_cast<int64_t>(no_rows_) * no_columns_);
}

float& Matrix::operator() (int index_row, int index_col) {
  return data_[static_cast<int64_t>(index_row) * no_columns_ + index_col];
}

const float& Matrix::operator() (int index_row, int index_col) const {
  return data_[static_cast<int64_t>(index_row) * no_columns_ + index_col];
}

void Matrix::resize(int no_rows, int no_columns) {
  no_rows_ = no_rows;
  no_columns_ = no_columns;
  if (no_rows > 0 && no_columns > 0) {
    data_.resize(static_cast<int64_t>(no_rows_) * no_columns_);
  }
}

int Matrix::no_rows() const { return no_rows_; }
int Matrix::no_columns() const { return no_columns_; }

float* Matrix::get_data() {
  return &data_[0];
}

const float* Matrix::get_data() const {
  return &data_[0];
}

}  // namespace artm
