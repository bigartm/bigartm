// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CPP_INTERFACE_H_
#define SRC_ARTM_CPP_INTERFACE_H_

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/c_interface.h"

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&);   \
  void operator=(const TypeName&)

#ifndef ARTM_ERROR_CODES_EXIST
#define ARTM_ERROR_CODES_EXIST
enum ArtmErrorCodes {
  ARTM_SUCCESS = 0,                   // Has no corresponding exception type.
  ARTM_STILL_WORKING = -1,            // Has no corresponding exception type.
  ARTM_INTERNAL_ERROR = -2,
  ARTM_ARGUMENT_OUT_OF_RANGE = -3,
  ARTM_INVALID_MASTER_ID = -4,
  ARTM_CORRUPTED_MESSAGE = -5,
  ARTM_INVALID_OPERATION = -6,
  ARTM_DISK_READ_ERROR = -7,
  ARTM_DISK_WRITE_ERROR = -8,
};
#endif

namespace artm {

class MasterComponent;
class Model;
class Regularizer;
class Dictionary;
class ProcessBatchesResultObject;

// Exception handling in cpp_interface
#define DEFINE_EXCEPTION_TYPE(Type, BaseType)                  \
class Type : public BaseType { public:  /*NOLINT*/             \
  Type() : BaseType("") {}                                     \
  explicit Type(std::string message) : BaseType(message) {}    \
};

DEFINE_EXCEPTION_TYPE(InternalError, std::runtime_error);
DEFINE_EXCEPTION_TYPE(ArgumentOutOfRangeException, std::runtime_error);
DEFINE_EXCEPTION_TYPE(InvalidMasterIdException, std::runtime_error);
DEFINE_EXCEPTION_TYPE(CorruptedMessageException, std::runtime_error);
DEFINE_EXCEPTION_TYPE(InvalidOperationException, std::runtime_error);
DEFINE_EXCEPTION_TYPE(DiskReadException, std::runtime_error);
DEFINE_EXCEPTION_TYPE(DiskWriteException, std::runtime_error);

#undef DEFINE_EXCEPTION_TYPE

void SaveBatch(const Batch& batch, const std::string& disk_path);
std::shared_ptr<Batch> LoadBatch(const std::string& filename);
void ParseCollection(const CollectionParserConfig& config);
void ConfigureLogging(const ConfigureLoggingArgs& args);

class Matrix {
 public:
  explicit Matrix(int no_rows = 0, int no_columns = 0) : no_rows_(no_rows), no_columns_(no_columns), data_() {
    if (no_rows > 0 && no_columns > 0)
      data_.resize(no_rows_ * no_columns_);
  }

  float& operator() (int index_row, int index_col) {
    return data_[index_row * no_columns_ + index_col];
  }

  const float& operator() (int index_row, int index_col) const {
    return data_[index_row * no_columns_ + index_col];
  }

  void resize(int no_rows, int no_columns) {
    no_rows_ = no_rows;
    no_columns_ = no_columns;
    if (no_rows > 0 && no_columns > 0)
      data_.resize(no_rows_ * no_columns_);
  }

  int no_rows() const { return no_rows_; }
  int no_columns() const { return no_columns_; }

  float* get_data() {
    return &data_[0];
  }

  const float* get_data() const {
    return &data_[0];
  }

 private:
  int no_rows_;
  int no_columns_;
  std::vector<float> data_;
  DISALLOW_COPY_AND_ASSIGN(Matrix);
};

class MasterComponent {
 public:
  explicit MasterComponent(const MasterComponentConfig& config);
  MasterComponent(const MasterComponent& rhs);
  ~MasterComponent();

  int id() const { return id_; }
  std::shared_ptr<TopicModel> GetTopicModel(const std::string& model_name);
  std::shared_ptr<TopicModel> GetTopicModel(const std::string& model_name, Matrix* matrix);
  std::shared_ptr<TopicModel> GetTopicModel(const GetTopicModelArgs& args);
  std::shared_ptr<TopicModel> GetTopicModel(const GetTopicModelArgs& args, Matrix* matrix);

  std::shared_ptr<Matrix> AttachTopicModel(const std::string& model_name);
  std::shared_ptr<Matrix> AttachTopicModel(const std::string& model_name, TopicModel* topic_model);

  std::shared_ptr<RegularizerInternalState> GetRegularizerState(
    const std::string& regularizer_name);
  std::shared_ptr<ThetaMatrix> GetThetaMatrix(const std::string& model_name);
  std::shared_ptr<ThetaMatrix> GetThetaMatrix(const std::string& model_name, Matrix* matrix);
  std::shared_ptr<ThetaMatrix> GetThetaMatrix(const std::string& model_name, const ::artm::Batch& batch);
  std::shared_ptr<ThetaMatrix> GetThetaMatrix(const GetThetaMatrixArgs& args);
  std::shared_ptr<ThetaMatrix> GetThetaMatrix(const GetThetaMatrixArgs& args, Matrix* matrix);
  std::shared_ptr<ScoreData> GetScore(const GetScoreValueArgs& args);

  template <typename T>
  std::shared_ptr<T> GetScoreAs(const std::string& model_name, const std::string& score_name);
  template <typename T>
  std::shared_ptr<T> GetScoreAs(const Model& model, const std::string& score_name);

  std::shared_ptr<MasterComponentInfo> info() const;

  std::shared_ptr<ProcessBatchesResultObject> ProcessBatches(const ProcessBatchesArgs& args);
  int AsyncProcessBatches(const ProcessBatchesArgs& args);
  int AwaitOperation(int operation_id);

  void MergeModel(const MergeModelArgs& args);
  void NormalizeModel(const NormalizeModelArgs& args);
  void RegularizeModel(const RegularizeModelArgs& args);
  void InitializeModel(const InitializeModelArgs& args);
  void ExportModel(const ExportModelArgs& args);
  void ImportModel(const ImportModelArgs& args);
  void DisposeModel(const std::string& model_name);

  void ImportBatches(const ImportBatchesArgs& args);
  void DisposeBatch(const std::string& batch_name);

  void CreateDictionary(const DictionaryData& args);
  void DisposeDictionary(const std::string& dictionary_name);
  void ImportDictionary(const ImportDictionaryArgs& args);
  void ExportDictionary(const ExportDictionaryArgs& args);
  void GatherDictionary(const GatherDictionaryArgs& args);
  void FilterDictionary(const FilterDictionaryArgs& args);
  std::shared_ptr<DictionaryData> GetDictionary(const std::string& dictionary_name);

  void Reconfigure(const MasterComponentConfig& config);
  bool AddBatch(const Batch& batch);
  bool AddBatch(const Batch& batch, bool reset_scores);
  bool AddBatch(const AddBatchArgs& args);
  void AddStream(const Stream& stream);
  void RemoveStream(std::string stream_name);

  void InvokeIteration(int iterations_count = 1);
  void InvokeIteration(const InvokeIterationArgs& args);

  bool WaitIdle(int timeout = -1);
  bool WaitIdle(const WaitIdleArgs& args);

  const MasterComponentConfig& config() const { return config_; }
  MasterComponentConfig* mutable_config() { return &config_; }

 private:
  int id_;
  MasterComponentConfig config_;
  MasterComponent& operator=(const MasterComponent&);
};

class MasterModel {
 public:
  explicit MasterModel(const MasterModelConfig& config);
  ~MasterModel();

  int id() const { return id_; }
  MasterComponentInfo info() const;  // misc. diagnostics information

  const MasterModelConfig& config() const { return config_; }
  MasterModelConfig* mutable_config() { return &config_; }
  void Reconfigure();  // apply MasterModel::config()

  // Operations to work with dictionary through disk
  void GatherDictionary(const GatherDictionaryArgs& args);
  void FilterDictionary(const FilterDictionaryArgs& args);
  void ImportDictionary(const ImportDictionaryArgs& args);
  void ExportDictionary(const ExportDictionaryArgs& args);
  void DisposeDictionary(const std::string& dictionary_name);

  // Operations to work with dictinoary through memory
  void CreateDictionary(const DictionaryData& args);
  DictionaryData GetDictionary(const GetDictionaryArgs& args);

  // Operatinos to work with batches through memory
  void ImportBatches(const ImportBatchesArgs& args);
  void DisposeBatch(const std::string& batch_name);

  // Operations to work with model
  void InitializeModel(const InitializeModelArgs& args);
  void OverwriteModel(const TopicModel& args);
  void ImportModel(const ImportModelArgs& args);
  void ExportModel(const ExportModelArgs& args);
  void FitOnlineModel(const FitOnlineMasterModelArgs& args);
  void FitOfflineModel(const FitOfflineMasterModelArgs& args);

  // Apply model to batches
  ThetaMatrix Transform(const TransformMasterModelArgs& args);
  ThetaMatrix Transform(const TransformMasterModelArgs& args, Matrix* matrix);

  // Retrieve operations
  TopicModel GetTopicModel(const GetTopicModelArgs& args);
  TopicModel GetTopicModel(const GetTopicModelArgs& args, Matrix* matrix);
  ThetaMatrix GetThetaMatrix(const GetThetaMatrixArgs& args);
  ThetaMatrix GetThetaMatrix(const GetThetaMatrixArgs& args, Matrix* matrix);

  // Retrieve scores
  ScoreData GetScore(const GetScoreValueArgs& args);
  template <typename T>
  T GetScoreAs(const GetScoreValueArgs& args);

 private:
  int id_;
  MasterModelConfig config_;

  MasterModel(const MasterModel& rhs);
  MasterModelConfig& operator=(const MasterModel&);
};

class Model {
 public:
  Model(const MasterComponent& master_component, const ModelConfig& config);
  ~Model();

  void Reconfigure(const ModelConfig& config);
  void Overwrite(const TopicModel& topic_model);
  void Overwrite(const TopicModel& topic_model, bool commit);
  void Initialize(const std::string& dictionary_name);
  void Export(const std::string& file_name);
  void Import(const std::string& file_name);
  void Enable();
  void Disable();
  void Synchronize(double decay);
  void Synchronize(double decay, double apply, bool invoke_regularizers);
  void Synchronize(const SynchronizeModelArgs& args);

  int master_id() const { return master_id_; }
  const std::string& name() const { return config_.name(); }

  const ModelConfig& config() const { return config_; }
  ModelConfig* mutable_config() { return &config_; }

 private:
  int master_id_;
  ModelConfig config_;
  DISALLOW_COPY_AND_ASSIGN(Model);
};

class Regularizer {
 public:
  Regularizer(const MasterComponent& master_component, const RegularizerConfig& config);
  ~Regularizer();

  void Reconfigure(const RegularizerConfig& config);

  int master_id() const { return master_id_; }
  const RegularizerConfig& config() const { return config_; }
  RegularizerConfig* mutable_config() { return &config_; }

 private:
  int master_id_;
  RegularizerConfig config_;
  DISALLOW_COPY_AND_ASSIGN(Regularizer);
};

template <typename T>
std::shared_ptr<T> MasterComponent::GetScoreAs(const std::string& model_name,
                                               const std::string& score_name) {
  GetScoreValueArgs args;
  args.set_model_name(model_name.c_str());
  args.set_score_name(score_name.c_str());
  auto score_data = GetScore(args);
  auto score = std::make_shared<T>();
  score->ParseFromString(score_data->data());
  return score;
}

template <typename T>
std::shared_ptr<T> MasterComponent::GetScoreAs(const Model& model,
                                               const std::string& score_name) {
  return GetScoreAs<T>(model.name(), score_name);
}
class ProcessBatchesResultObject {
 public:
  explicit ProcessBatchesResultObject(ProcessBatchesResult message) : message_(message) {}

  template <typename T>
  std::shared_ptr<T> GetScoreAs(const std::string& score_name) const;
  inline const ThetaMatrix& GetThetaMatrix() const { return message_.theta_matrix(); }

 private:
  ProcessBatchesResult message_;
};

template <typename T>
std::shared_ptr<T> ProcessBatchesResultObject::GetScoreAs(const std::string& score_name) const {
  for (int i = 0; i < message_.score_data_size(); ++i) {
    if (message_.score_data(i).name() == score_name) {
      auto score = std::make_shared<T>();
      score->ParseFromString(message_.score_data(i).data());
      return score;
    }
  }

  return nullptr;
}

template <typename T>
T MasterModel::GetScoreAs(const GetScoreValueArgs& args) {
  auto score_data = GetScore(args);
  T score;
  score.ParseFromString(score_data.data());
  return score;
}

}  // namespace artm

#endif  // SRC_ARTM_CPP_INTERFACE_H_
