// Copyright 2017, Additive Regularization of Topic Models.

#include "artm/core/master_component.h"

#include <algorithm>
#include <fstream>  // NOLINT
#include <vector>
#include <unordered_set>
#include <sstream>
#include <utility>

#include "boost/algorithm/string.hpp"
#include "boost/algorithm/string/predicate.hpp"
#include "boost/thread.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"
#include "boost/uuid/random_generator.hpp"
#include "boost/lexical_cast.hpp"

#include "glog/logging.h"

#include "artm/regularizer_interface.h"
#include "artm/score_calculator_interface.h"

#include "artm/core/dictionary_operations.h"
#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/batch_manager.h"
#include "artm/core/cache_manager.h"
#include "artm/core/call_on_destruction.h"
#include "artm/core/check_messages.h"
#include "artm/core/instance.h"
#include "artm/core/processor.h"
#include "artm/core/protobuf_helpers.h"
#include "artm/core/phi_matrix_operations.h"
#include "artm/core/score_manager.h"
#include "artm/core/dense_phi_matrix.h"
#include "artm/core/template_manager.h"

typedef artm::core::TemplateManager<std::shared_ptr< ::artm::core::MasterComponent>> MasterComponentManager;

namespace artm {
namespace core {

static void HandleExternalTopicModelRequest(::artm::TopicModel* topic_model, std::string* lm) {
  lm->resize(sizeof(float) * topic_model->token_size() * topic_model->num_topics());
  char* lm_ptr = &(*lm)[0];
  float* lm_float = reinterpret_cast<float*>(lm_ptr);
  for (int64_t token_index = 0; token_index < topic_model->token_size(); ++token_index) {
    for (int64_t topic_index = 0; topic_index < topic_model->num_topics(); ++topic_index) {
      int64_t index = token_index * topic_model->num_topics() + topic_index;
      lm_float[index] = topic_model->token_weights(token_index).value(topic_index);
    }
  }

  topic_model->clear_token_weights();
}

static void HandleExternalThetaMatrixRequest(::artm::ThetaMatrix* theta_matrix, std::string* lm) {
  lm->resize(sizeof(float) * theta_matrix->item_id_size() * theta_matrix->num_topics());
  char* lm_ptr = &(*lm)[0];
  float* lm_float = reinterpret_cast<float*>(lm_ptr);
  for (int64_t topic_index = 0; topic_index < theta_matrix->num_topics(); ++topic_index) {
    for (int64_t item_index = 0; item_index < theta_matrix->item_id_size(); ++item_index) {
      int64_t index = item_index * theta_matrix->num_topics() + topic_index;
      lm_float[index] = theta_matrix->item_weights(item_index).value(topic_index);
    }
  }

  theta_matrix->clear_item_weights();
}

static void HandleSparseTopicModelRequest(::artm::TopicModel* topic_model, std::string* lm) {
  if (topic_model->topic_indices_size() == 0) {
    BOOST_THROW_EXCEPTION(InternalError("topic_model->topic_indices_size() == 0"));
  }

  int32_t coo_token = 0;
  std::vector<int32_t> coo_token_index;
  std::vector<int32_t> coo_topic_index;
  std::vector<float> coo_weight;
  ::google::protobuf::RepeatedPtrField<std::string> token;
  ::google::protobuf::RepeatedPtrField<std::string> class_id;

  for (int token_index = 0; token_index < topic_model->token_weights_size(); ++token_index) {
    const artm::IntArray& topic_indices = topic_model->topic_indices(token_index);
    const artm::FloatArray& token_weights = topic_model->token_weights(token_index);
    if (token_weights.value_size() == 0) {
      continue;
    }

    for (int value_index = 0; value_index < topic_indices.value_size(); ++value_index) {
      coo_token_index.push_back(coo_token);
      coo_topic_index.push_back(topic_indices.value(value_index));
      coo_weight.push_back(token_weights.value(value_index));
    }

    coo_token++;
    if (topic_model->token_size() > 0) {
      token.Add()->assign(topic_model->token(token_index));
    }
    if (topic_model->class_id_size() > 0) {
      class_id.Add()->assign(topic_model->class_id(token_index));
    }
  }

  if (coo_weight.empty()) {
    BOOST_THROW_EXCEPTION(InvalidOperation("No data to return for sparse phi matrix"));
  }

  int64_t num_values = coo_weight.size();
  int64_t byte_size = sizeof(int32_t) * num_values;  // assert(sizeof(float) == sizeof(int32_t)
  lm->resize(3 * byte_size);
  memcpy(&(*lm)[0], &coo_token_index[0], byte_size);
  memcpy(&(*lm)[byte_size], &coo_topic_index[0], byte_size);
  memcpy(&(*lm)[2 * byte_size], &coo_weight[0], byte_size);
  topic_model->mutable_token()->Swap(&token);
  topic_model->mutable_class_id()->Swap(&class_id);
  topic_model->clear_token_weights();
  topic_model->clear_topic_indices();
  topic_model->set_num_values(num_values);
}

static void HandleSparseThetaMatrixRequest(::artm::ThetaMatrix* theta_matrix, std::string* lm) {
  if (theta_matrix->topic_indices_size() == 0) {
    BOOST_THROW_EXCEPTION(InternalError("theta_matrix->topic_indices_size() == 0"));
  }

  int32_t coo_item = 0;
  std::vector<int32_t> coo_item_index;
  std::vector<int32_t> coo_topic_index;
  std::vector<float> coo_weight;
  ::google::protobuf::RepeatedField< ::google::protobuf::int32> item_id;
  ::google::protobuf::RepeatedPtrField<std::string> item_title;

  for (int item_index = 0; item_index < theta_matrix->item_weights_size(); ++item_index) {
    const artm::IntArray& topic_indices = theta_matrix->topic_indices(item_index);
    const artm::FloatArray& item_weights = theta_matrix->item_weights(item_index);
    if (item_weights.value_size() == 0) {
      continue;
    }

    for (int value_index = 0; value_index < topic_indices.value_size(); ++value_index) {
      coo_item_index.push_back(coo_item);
      coo_topic_index.push_back(topic_indices.value(value_index));
      coo_weight.push_back(item_weights.value(value_index));
    }

    coo_item++;
    if (theta_matrix->item_id_size() > 0) {
      item_id.Add(theta_matrix->item_id(item_index));
    }
    if (theta_matrix->item_title_size() > 0) {
      item_title.Add()->assign(theta_matrix->item_title(item_index));
    }
  }

  if (coo_weight.empty()) {
    BOOST_THROW_EXCEPTION(InvalidOperation("No data to return for sparse theta matrix"));
  }

  int64_t num_values = coo_weight.size();
  int64_t byte_size = sizeof(int32_t) * num_values;  // assert(sizeof(float) == sizeof(int32_t)
  lm->resize(3 * byte_size);
  memcpy(&(*lm)[0], &coo_item_index[0], byte_size);
  memcpy(&(*lm)[byte_size], &coo_topic_index[0], byte_size);
  memcpy(&(*lm)[2 * byte_size], &coo_weight[0], byte_size);
  theta_matrix->mutable_item_id()->Swap(&item_id);
  theta_matrix->mutable_item_title()->Swap(&item_title);
  theta_matrix->clear_item_weights();
  theta_matrix->clear_topic_indices();
  theta_matrix->set_num_values(num_values);
}

void MasterComponent::CreateOrReconfigureMasterComponent(const MasterModelConfig& config,
                                                         bool reconfigure,
                                                         bool change_topic_name) {
  if (!reconfigure) {
    instance_ = std::make_shared<Instance>(config);
  } else {
    auto old_config = instance_->config();
    if (!change_topic_name && (old_config->topic_name_size() != config.topic_name_size())) {
      BOOST_THROW_EXCEPTION(InvalidOperation(
        "ArtmReconfigureMasterModel can not change number of topics; use ArtmReconfigureTopicName"));
    }
    if (old_config->ptd_name() != config.ptd_name()) {
      BOOST_THROW_EXCEPTION(InvalidOperation(
        "ArtmReconfigureMasterModel can not change MasterModelConfig.ptd_name"));
    }

    instance_->Reconfigure(config);

    // If there is a change in config.topic_name, update all phi matrices to new set of topics
    // 1. If change_topic_names is true:
    //    Topics that were removed from the config will be also removed from phi matrices.
    //    New topics will be initialized with zeros.
    // 2. If change_topic_names is false:
    //    Topic names will be set as new labels.
    if (!repeated_field_equals(old_config->topic_name(), config.topic_name())) {
      auto model_names = instance_->models()->keys();
      for (const auto& model_name : model_names) {
        auto model = instance_->GetPhiMatrixSafe(model_name);

        // Ignore models where set of topics doesn't match the old config
        if (!repeated_field_equals(model->topic_name(), old_config->topic_name())) {
          continue;
        }

        if (change_topic_name) {
          MergeModelArgs merge;
          merge.add_nwt_source_name(model_name);
          merge.add_source_weight(1.0f);
          merge.set_nwt_target_name(model_name);
          MergeModel(merge);
        } else {
          for (int topic_index = 0; topic_index < config.topic_name_size(); topic_index++) {
            (const_cast<PhiMatrix*>(model.get()))->set_topic_name(topic_index, config.topic_name(topic_index));
          }
        }
      }
    }
  }

  if (reconfigure) {  // remove all regularizers
    instance_->regularizers()->clear();
  }

  // create (or re-create the regularizers)
  for (int i = 0; i < config.regularizer_config_size(); ++i) {
    CreateOrReconfigureRegularizer(config.regularizer_config(i));
  }
}

MasterComponent::MasterComponent(const MasterModelConfig& config)
    : instance_(nullptr) {
  CreateOrReconfigureMasterComponent(config, /*reconfigure =*/ false, /*change_topic_name*/ false);
}

MasterComponent::MasterComponent(const MasterComponent& rhs)
    : instance_(rhs.instance_->Duplicate()) {
}

MasterComponent::~MasterComponent() { }

std::shared_ptr<MasterComponent> MasterComponent::Duplicate() const {
  return std::shared_ptr<MasterComponent>(new MasterComponent(*this));
}

std::shared_ptr<MasterModelConfig> MasterComponent::config() const {
  return instance_->config();
}

void MasterComponent::DisposeModel(const std::string& name) {
  instance_->DisposeModel(name);
}

void MasterComponent::ClearThetaCache(const ClearThetaCacheArgs& args) {
  instance_->cache_manager()->Clear();
}

void MasterComponent::ClearScoreCache(const ClearScoreCacheArgs& args) {
  instance_->score_manager()->Clear();
}

void MasterComponent::ClearScoreArrayCache(const ClearScoreArrayCacheArgs& args) {
  instance_->score_tracker()->Clear();
}

void MasterComponent::CreateOrReconfigureRegularizer(const RegularizerConfig& config) {
  instance_->CreateOrReconfigureRegularizer(config);
}

void MasterComponent::DisposeRegularizer(const std::string& name) {
  instance_->DisposeRegularizer(name);
}

void MasterComponent::AddDictionary(std::shared_ptr<Dictionary> dictionary) {
  DisposeDictionary(dictionary->name());
  instance_->dictionaries()->set(dictionary->name(), dictionary);
  DictionaryOperations::WriteDictionarySummaryToLog(*dictionary);
}

void MasterComponent::CreateDictionary(const DictionaryData& data) {
  AddDictionary(DictionaryOperations::Create(data));
}

void MasterComponent::DisposeDictionary(const std::string& name) {
  if (name.empty()) {
    instance_->dictionaries()->clear();
  } else {
    instance_->dictionaries()->erase(name);
  }
}

void MasterComponent::ExportDictionary(const ExportDictionaryArgs& args) {
  std::shared_ptr<Dictionary> dict_ptr = instance_->dictionaries()->get(args.dictionary_name());
  if (dict_ptr == nullptr) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Dictionary " +
        args.dictionary_name() + " does not exist or has no tokens"));
  }

  DictionaryOperations::Export(args, *dict_ptr);
}

void MasterComponent::ImportDictionary(const ImportDictionaryArgs& args) {
  auto dictionary = DictionaryOperations::Import(args);
  AddDictionary(dictionary);
  LOG(INFO) << "Import completed, token_size = " << dictionary->size();
}

void MasterComponent::Request(::artm::MasterModelConfig* result) {
  std::shared_ptr<MasterModelConfig> config = instance_->config();
  if (config == nullptr) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
        "Invalid master_id; use ArtmCreateMasterModel instead of ArtmCreateMasterComponent"));
  }

  result->CopyFrom(*config);
}

void MasterComponent::Request(const GetDictionaryArgs& args, DictionaryData* result) {
  std::shared_ptr<Dictionary> dict_ptr = instance_->dictionaries()->get(args.dictionary_name());
  if (dict_ptr == nullptr) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Dictionary " +
        args.dictionary_name() + " does not exist or has no tokens"));
  }
  DictionaryOperations::StoreIntoDictionaryData(*dict_ptr, result);
  result->set_name(args.dictionary_name());
}

void MasterComponent::ImportBatches(const ImportBatchesArgs& args) {
  std::shared_ptr<MasterModelConfig> config = instance_->config();
  if (config == nullptr) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Invalid master_id"));
  }

  for (int i = 0; i < args.batch_size(); ++i) {
    std::shared_ptr<Batch> batch = std::make_shared<Batch>(args.batch(i));
    if ((batch->description() == ::artm::core::kParentPhiMatrixBatch) && (batch->item_size() == 0)) {
      LOG(INFO) << "Retrieving batch (id=" << batch->id()
                << ") from parent master model (id = " << config->parent_master_model_id() << ")";
      auto parent_master = MasterComponentManager::singleton().Get(config->parent_master_model_id());
      if (parent_master == nullptr || parent_master->config() == nullptr) {
        BOOST_THROW_EXCEPTION(InvalidOperation(
          "Unable to access parent master component with given id "
          "(MasterComponentConfig.parent_master_model_id)"));
      }

      // Get nwt matrix from parent master component
      ::artm::GetTopicModelArgs get_topic_model_args;
      get_topic_model_args.mutable_class_id()->CopyFrom(config->class_id());
      get_topic_model_args.set_matrix_layout(MatrixLayout_Sparse);
      get_topic_model_args.set_model_name(parent_master->config()->nwt_name());
      FixMessage(&get_topic_model_args);
      ::artm::TopicModel topic_model;
      parent_master->Request(get_topic_model_args, &topic_model);

      PhiMatrixOperations::ConvertTopicModelToPseudoBatch(&topic_model, batch.get());
    }
    FixAndValidateMessage(batch.get(), /* throw_error =*/ true);
    instance_->batches()->set(batch->id(), batch);
  }
}

void MasterComponent::DisposeBatch(const std::string& name) {
  instance_->batches()->erase(name);
}

void MasterComponent::ExportModel(const ExportModelArgs& args) {
  std::shared_ptr<MasterModelConfig> config = instance_->config();
  if (config != nullptr) {
    if (!args.has_model_name()) {
      const_cast<ExportModelArgs*>(&args)->set_model_name(config->pwt_name());
    }
  }

  if (boost::filesystem::exists(args.file_name())) {
    BOOST_THROW_EXCEPTION(DiskWriteException("File already exists: " + args.file_name()));
  }

  std::ofstream fout(args.file_name(), std::ofstream::binary);
  if (!fout.is_open()) {
    BOOST_THROW_EXCEPTION(DiskWriteException("Unable to create file " + args.file_name()));
  }

  std::shared_ptr<const PhiMatrix> phi_matrix = instance_->GetPhiMatrixSafe(args.model_name());
  const PhiMatrix& n_wt = *phi_matrix;

  LOG(INFO) << "Exporting model " << args.model_name() << " to " << args.file_name();

  const int token_size = n_wt.token_size();
  if (token_size == 0) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Model " + args.model_name() + " has no tokens, export failed"));
  }

  int tokens_per_chunk = std::min<int>(token_size, 100 * 1024 * 1024 / n_wt.topic_size());

  ::artm::GetTopicModelArgs get_topic_model_args;
  get_topic_model_args.set_model_name(args.model_name());
  get_topic_model_args.set_matrix_layout(::artm::MatrixLayout_Sparse);
  get_topic_model_args.mutable_token()->Reserve(tokens_per_chunk);
  get_topic_model_args.mutable_class_id()->Reserve(tokens_per_chunk);

  const char version = 0;
  fout << version;

  for (int token_id = 0; token_id < token_size; ++token_id) {
    Token token = n_wt.token(token_id);
    get_topic_model_args.add_token(token.keyword);
    get_topic_model_args.add_class_id(token.class_id);

    if (((token_id + 1) == token_size) || (get_topic_model_args.token_size() >= tokens_per_chunk)) {
      ::artm::TopicModel external_topic_model;
      PhiMatrixOperations::RetrieveExternalTopicModel(n_wt, get_topic_model_args, &external_topic_model);
      std::string str = external_topic_model.SerializeAsString();
      if (str.size() >= kProtobufCodedStreamTotalBytesLimit) {
        BOOST_THROW_EXCEPTION(InvalidOperation("TopicModel is too large to export"));
      }
      fout << str.size();
      fout << str;
      get_topic_model_args.clear_class_id();
      get_topic_model_args.clear_token();
    }
  }

  fout.close();
  LOG(INFO) << "Export of model completed, token_size = " << n_wt.token_size()
            << ", topic_size = " << n_wt.topic_size();
}

void MasterComponent::ImportModel(const ImportModelArgs& args) {
  std::shared_ptr<MasterModelConfig> config = instance_->config();
  if (config != nullptr) {
    if (!args.has_model_name()) {
      const_cast<ImportModelArgs*>(&args)->set_model_name(config->pwt_name());
    }
  }

  std::ifstream fin(args.file_name(), std::ifstream::binary);
  if (!fin.is_open()) {
    BOOST_THROW_EXCEPTION(DiskReadException("Unable to open file " + args.file_name()));
  }

  LOG(INFO) << "Importing model " << args.model_name() << " from " << args.file_name();

  char version;
  fin >> version;
  if (version != 0) {
    std::stringstream ss;
    ss << "Unsupported format version: " << static_cast<int>(version);
    BOOST_THROW_EXCEPTION(DiskReadException(ss.str()));
  }

  std::shared_ptr<DensePhiMatrix> target = nullptr;
  while (!fin.eof()) {
    int length;
    fin >> length;
    if (fin.eof()) {
      break;
    }

    if (length <= 0) {
      BOOST_THROW_EXCEPTION(CorruptedMessageException(
          "Unable to read from " + args.file_name() + ": message has negative length"));
    }

    std::string buffer(length, '\0');
    fin.read(&buffer[0], length);
    ::artm::TopicModel topic_model;
    if (!topic_model.ParseFromArray(buffer.c_str(), length)) {
      BOOST_THROW_EXCEPTION(CorruptedMessageException(
          "Unable to read from " + args.file_name() + ": message parsing failed"));
    }

    topic_model.set_name(args.model_name());
    target = std::make_shared<DensePhiMatrix>(args.model_name(), topic_model.topic_name());

    PhiMatrixOperations::ApplyTopicModelOperation(topic_model, 1.0f, /* add_missing_tokens = */ true, target.get());
  }

  fin.close();

  if (target == nullptr) {
    BOOST_THROW_EXCEPTION(CorruptedMessageException("Unable to read from " + args.file_name()));
  }

  instance_->SetPhiMatrix(args.model_name(), target);
  LOG(INFO) << "Import of model completed, token_size = " << target->token_size()
            << ", topic_size = " << target->topic_size();
}

void MasterComponent::ExportScoreTracker(const ExportScoreTrackerArgs& args) {
  if (boost::filesystem::exists(args.file_name())) {
    BOOST_THROW_EXCEPTION(DiskWriteException("File already exists: " + args.file_name()));
  }

  std::ofstream fout(args.file_name(), std::ofstream::binary);
  if (!fout.is_open()) {
    BOOST_THROW_EXCEPTION(DiskWriteException("Unable to create file " + args.file_name()));
  }

  LOG(INFO) << "Exporting score tracker to " << args.file_name();

  const char version = 0;
  fout << version;

  // We expect here that each ScoreData object has suitable size (< 2GB)
  for (auto& item : instance_->score_tracker()->GetDataUnsafe()) {
    auto str = item->SerializeAsString();
    if (str.size() >= kProtobufCodedStreamTotalBytesLimit) {
      BOOST_THROW_EXCEPTION(InvalidOperation("ScoreTracker is too large to export"));
    }
    fout << str.size();
    fout << str;
  }

  fout.close();
  LOG(INFO) << "Export of score tracker completed, number of score items: " << instance_->score_tracker()->Size();
}

void MasterComponent::ImportScoreTracker(const ImportScoreTrackerArgs& args) {
  std::ifstream fin(args.file_name(), std::ifstream::binary);
  if (!fin.is_open()) {
    BOOST_THROW_EXCEPTION(DiskReadException("Unable to open file " + args.file_name()));
  }

  LOG(INFO) << "Importing score tracker from " << args.file_name();

  char version;
  fin >> version;
  if (version != 0) {
    std::stringstream ss;
    ss << "Unsupported format version: " << static_cast<int>(version);
    BOOST_THROW_EXCEPTION(DiskReadException(ss.str()));
  }

  instance_->score_tracker()->Clear();
  while (!fin.eof()) {
    int length;
    fin >> length;
    if (fin.eof()) {
      break;
    }

    if (length <= 0) {
      BOOST_THROW_EXCEPTION(CorruptedMessageException(
          "Unable to read from " + args.file_name() + ": message has negative length"));
    }

    std::string buffer(length, '\0');
    fin.read(&buffer[0], length);
    ::artm::ScoreData score_data;
    if (!score_data.ParseFromArray(buffer.c_str(), length)) {
      BOOST_THROW_EXCEPTION(CorruptedMessageException(
          "Unable to read from " + args.file_name() + ": message parsing failed"));
    }

    auto ptr = instance_->score_tracker()->Add();
    ptr->CopyFrom(score_data);
  }

  fin.close();

  LOG(INFO) << "Import of model completed, number of score items: " << instance_->score_tracker()->Size();
}

void MasterComponent::AttachModel(const AttachModelArgs& args, int address_length, float* address) {
  ModelName model_name = args.model_name();
  LOG(INFO) << "Attaching model " << model_name << " to " << address << " (" << address_length << " bytes)";

  std::shared_ptr<const PhiMatrix> phi_matrix = instance_->GetPhiMatrixSafe(model_name);

  PhiMatrixFrame* frame = dynamic_cast<PhiMatrixFrame*>(const_cast<PhiMatrix*>(phi_matrix.get()));
  if (frame == nullptr) {
    BOOST_THROW_EXCEPTION(InvalidOperation("Unable to attach to model " + model_name));
  }

  std::shared_ptr<AttachedPhiMatrix> attached = std::make_shared<AttachedPhiMatrix>(address_length, address, frame);
  instance_->SetPhiMatrix(model_name, attached);
}

void MasterComponent::InitializeModel(const InitializeModelArgs& args) {
  std::shared_ptr<MasterModelConfig> config = instance_->config();
  if (config != nullptr) {
    InitializeModelArgs* mutable_args = const_cast<InitializeModelArgs*>(&args);
    if (!args.has_model_name()) {
      mutable_args->set_model_name(config->pwt_name());
    }
    if (args.topic_name_size() == 0) {
      mutable_args->mutable_topic_name()->CopyFrom(config->topic_name());
    }
    FixMessage(mutable_args);

    if (config->has_parent_master_model_id() && (!args.has_seed() || args.seed() == -1)) {
      BOOST_THROW_EXCEPTION(InvalidOperation(
        "InitializeModelArgs.seed must be specified for hARTM. "
        "This error happens because MasterModelConfig.parent_master_model_id is specified."));
    }
  }

  std::shared_ptr<PhiMatrix> new_ttm;
  int excluded_tokens = 0;
  if (args.has_dictionary_name()) {
    auto dict = instance_->dictionaries()->get(args.dictionary_name());
    if (dict == nullptr) {
      std::stringstream ss;
      ss << "Dictionary '" << args.dictionary_name() << "' does not exist";
      BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    }

    if (dict->size() == 0) {
      std::stringstream ss;
      ss << "Dictionary '" << args.dictionary_name() << "' has no entries";
      BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    }

    new_ttm = std::make_shared< ::artm::core::DensePhiMatrix>(args.model_name(), args.topic_name());
    for (int index = 0; index < (int64_t) dict->size(); ++index) {
      ::artm::core::Token token = dict->entry(index)->token();

      if (config->class_id_size() > 0 && !is_member(token.class_id, config->class_id())) {
        continue;
      }
      new_ttm->AddToken(token);
    }

    excluded_tokens = dict->size() - new_ttm->token_size();
    LOG_IF(INFO, excluded_tokens > 0)
      << excluded_tokens
      << " tokens were present in the dictionary, but excluded from the model";

  } else {
    // Initialize without dictionary from an existing model
    std::shared_ptr<const PhiMatrix> ttm = instance_->GetPhiMatrix(args.model_name());
    if (ttm == nullptr) {
      std::stringstream ss;
      ss << "Invalid usage of InitializeModelArgs --- ";
      ss << "InitializeModelArgs.dictionary is empty, and model '" << args.model_name() << "' does not exist; ";
      BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
    }

    new_ttm = ttm->Duplicate();
    PhiMatrixOperations::AssignValue(0.0f, new_ttm.get());
  }

  if (new_ttm->token_size() == 0) {
    std::stringstream ss;
    ss << "Unable to initialize matrix from " << args.dictionary_name() << " dictionary. ";
    ss << "Either the dictionary is empty, or no tokens in the dictionary matches class_id(s), ";
    ss << "listed in the configuration of the model";
    BOOST_THROW_EXCEPTION(InvalidOperation(ss.str()));
  }

  for (int token_index = 0; token_index < new_ttm->token_size(); token_index++) {
    Token token = new_ttm->token(token_index);
    std::vector<float> vec = Helpers::GenerateRandomVector(new_ttm->topic_size(), token, args.seed());
    new_ttm->increase(token_index, vec);
  }

  PhiMatrixOperations::FindPwt(*new_ttm, new_ttm.get());
  instance_->SetPhiMatrix(args.model_name(), new_ttm);

  LOG(INFO) << "InitializeModel() created matrix " << new_ttm->model_name()
            << " with " << args.topic_name_size() << " topics and "
            << new_ttm->token_size() << " tokens; ";
}

void MasterComponent::FilterDictionary(const FilterDictionaryArgs& args) {
  auto src_dictionary_ptr = instance_->dictionaries()->get(args.dictionary_name());
  if (src_dictionary_ptr == nullptr) {
    LOG(ERROR) << "Dictionary::Filter(): filter was requested for non-exists dictionary '"
               << args.dictionary_name() << "', operation was aborted";
    return;
  }

  AddDictionary(DictionaryOperations::Filter(args, *src_dictionary_ptr));
}

void MasterComponent::GatherDictionary(const GatherDictionaryArgs& args) {
  AddDictionary(DictionaryOperations::Gather(args, *instance_->batches()));
}

void MasterComponent::ReconfigureMasterModel(const MasterModelConfig& config) {
  CreateOrReconfigureMasterComponent(config,
                                     /*reconfigure = */ true,
                                     /*change_topic_name = */ false);
}

void MasterComponent::ReconfigureTopicName(const MasterModelConfig& config) {
  CreateOrReconfigureMasterComponent(config,
                                     /*reconfigure = */ true,
                                     /*change_topic_name = */ true);
}

void MasterComponent::Request(const GetTopicModelArgs& args, ::artm::TopicModel* result) {
  std::shared_ptr<MasterModelConfig> config = instance_->config();
  if (config != nullptr) {
    if (!args.has_model_name()) {
      const_cast<GetTopicModelArgs*>(&args)->set_model_name(config->pwt_name());
    }
  }

  auto phi_matrix = instance_->GetPhiMatrixSafe(args.model_name());
  PhiMatrixOperations::RetrieveExternalTopicModel(*phi_matrix, args, result);
}

void MasterComponent::Request(const GetTopicModelArgs& args, ::artm::TopicModel* result, std::string* external) {
  Request(args, result);
  if (args.matrix_layout() == artm::MatrixLayout_Sparse) {
    HandleSparseTopicModelRequest(result, external);
  } else {
    HandleExternalTopicModelRequest(result, external);
  }
}

void MasterComponent::Request(const GetScoreValueArgs& args, ScoreData* result) {
  instance_->score_manager()->RequestScore(args.score_name(), result);
}

void MasterComponent::Request(const GetScoreArrayArgs& args, ScoreArray* result) {
  instance_->score_tracker()->RequestScoreArray(args, result);
}

void MasterComponent::Request(const GetMasterComponentInfoArgs& /*args*/, MasterComponentInfo* result) {
  this->instance_->RequestMasterComponentInfo(result);
}

void MasterComponent::Request(const ProcessBatchesArgs& args, ProcessBatchesResult* result) {
  BatchManager batch_manager;
  RequestProcessBatchesImpl(args, &batch_manager, /* async =*/ false, nullptr, result->mutable_theta_matrix());
  instance_->score_manager()->RequestAllScores(result->mutable_score_data());
}

void MasterComponent::Request(const ProcessBatchesArgs& args, ProcessBatchesResult* result, std::string* external) {
  const bool is_dense_theta = args.theta_matrix_type() == artm::ThetaMatrixType_Dense;
  const bool is_dense_ptdw = args.theta_matrix_type() == artm::ThetaMatrixType_DensePtdw;
  const bool is_sparse_theta = args.theta_matrix_type() == artm::ThetaMatrixType_Sparse;
  if (!is_dense_theta && !is_dense_ptdw && !is_sparse_theta) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
        "Dense or Sparse matrix format is required for ArtmRequestProcessBatchesExternal"));
  }

  Request(args, result);
  if (is_sparse_theta) {
    HandleSparseThetaMatrixRequest(result->mutable_theta_matrix(), external);
  } else {
    HandleExternalThetaMatrixRequest(result->mutable_theta_matrix(), external);
  }
}

void MasterComponent::AsyncRequestProcessBatches(const ProcessBatchesArgs& process_batches_args,
                                                 BatchManager *batch_manager) {
  RequestProcessBatchesImpl(process_batches_args, batch_manager, /* async =*/ true,
                            /*score_manager=*/ nullptr, /* theta_matrix=*/ nullptr);
}

void MasterComponent::RequestProcessBatchesImpl(const ProcessBatchesArgs& process_batches_args,
                                                BatchManager* batch_manager, bool async,
                                                ScoreManager* score_manager,
                                                ::artm::ThetaMatrix* theta_matrix) {
  const ProcessBatchesArgs& args = process_batches_args;  // short notation
  ModelName model_name = args.pwt_source_name();

  if (instance_->processor_size() <= 0) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
        "Can't process batches because there are no processors. Check MasterModelConfig.num_processors setting."));
  }

  std::shared_ptr<const PhiMatrix> phi_matrix = instance_->GetPhiMatrixSafe(model_name);
  const PhiMatrix& p_wt = *phi_matrix;
  const_cast<ProcessBatchesArgs*>(&args)->mutable_topic_name()->CopyFrom(p_wt.topic_name());
  if (args.has_nwt_target_name()) {
    if (args.nwt_target_name() == args.pwt_source_name()) {
      BOOST_THROW_EXCEPTION(InvalidOperation(
          "ProcessBatchesArgs.pwt_source_name == ProcessBatchesArgs.nwt_target_name"));
    }

    // If nwt_target_name already exists, assign all its elements to zero.
    // Otherwise, create new n_wt matrix of the same shape as p_wt matrix.
    auto current_nwt_target = instance_->GetPhiMatrix(args.nwt_target_name());
    if (current_nwt_target != nullptr) {
      if (process_batches_args.reset_nwt()) {
        PhiMatrixOperations::AssignValue(0.0f, const_cast<::artm::core::PhiMatrix*>(current_nwt_target.get()));
      }
    } else {
      auto nwt_target(std::make_shared<DensePhiMatrix>(args.nwt_target_name(), p_wt.topic_name()));
      nwt_target->Reshape(p_wt);
      instance_->SetPhiMatrix(args.nwt_target_name(), nwt_target);
    }
  }

  if (async && args.theta_matrix_type() != ThetaMatrixType_None) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
        "ArtmAsyncProcessBatches require ProcessBatchesArgs.theta_matrix_type to be set to None"));
  }

  // The code below must not use cache_manger in async mode.
  // Since cache_manager lives on stack it will be destroyed once we return from this function.
  // Therefore, no pointers to cache_manager should exist upon return from RequestProcessBatchesImpl.
  CacheManager cache_manager("", nullptr);

  bool return_theta = false;
  bool return_ptdw = false;

  CacheManager* ptdw_cache_manager_ptr = nullptr;
  CacheManager* theta_cache_manager_ptr = nullptr;
  switch (args.theta_matrix_type()) {
    case ThetaMatrixType_Cache:
      if (instance_->config()->cache_theta()) {
        theta_cache_manager_ptr = instance_->cache_manager();
      }
      break;
    case ThetaMatrixType_Dense:
    case ThetaMatrixType_Sparse:
      theta_cache_manager_ptr = &cache_manager;
      return_theta = true;
      break;
    case ThetaMatrixType_DensePtdw:
    case ThetaMatrixType_SparsePtdw:
      ptdw_cache_manager_ptr = &cache_manager;
      return_ptdw = true;
  }

  if (args.batch_filename_size() < instance_->processor_size()) {
    LOG_FIRST_N(INFO, 1) << "Batches count (=" << args.batch_filename_size()
                         << ") is smaller than num processors (="
                         << instance_->processor_size()
                         << "), which may cause suboptimal performance.";
  }

  auto createProcessorInput = [&](){  // NOLINT
    boost::uuids::uuid task_id = boost::uuids::random_generator()();
    batch_manager->Add(task_id);

    auto pi = std::make_shared<ProcessorInput>();
    pi->set_batch_manager(batch_manager);
    pi->set_score_manager(score_manager);
    pi->set_cache_manager(theta_cache_manager_ptr);
    pi->set_ptdw_cache_manager(ptdw_cache_manager_ptr);
    pi->set_model_name(model_name);
    pi->mutable_args()->CopyFrom(args);
    pi->set_task_id(task_id);

    if (args.reuse_theta()) {
      pi->set_reuse_theta_cache_manager(instance_->cache_manager());
    }

    if (args.has_nwt_target_name()) {
      pi->set_nwt_target_name(args.nwt_target_name());
    }

    return pi;
  };

  // Enqueue tasks based on args.batch_filename
  for (int batch_index = 0; batch_index < args.batch_filename_size(); ++batch_index) {
    auto pi = createProcessorInput();
    pi->set_batch_filename(args.batch_filename(batch_index));
    pi->set_batch_weight(args.batch_weight(batch_index));
    instance_->processor_queue()->push(pi);
  }

  // Enqueue tasks based on args.batch
  for (int batch_index = 0; batch_index < args.batch_size(); ++batch_index) {
    auto pi = createProcessorInput();
    pi->mutable_batch()->CopyFrom(args.batch(batch_index));
    pi->set_batch_weight(args.batch_weight(batch_index));
    instance_->processor_queue()->push(pi);
  }

  if (async) {
    return;
  }

  while (!batch_manager->IsEverythingProcessed()) {
    boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));
  }

  GetThetaMatrixArgs get_theta_matrix_args;
  switch (args.theta_matrix_type()) {
    case ThetaMatrixType_Dense:
    case ThetaMatrixType_DensePtdw:
      get_theta_matrix_args.set_matrix_layout(MatrixLayout_Dense);
      break;
    case ThetaMatrixType_Sparse:
    case ThetaMatrixType_SparsePtdw:
      get_theta_matrix_args.set_matrix_layout(MatrixLayout_Sparse);
      break;
  }

  if (theta_matrix != nullptr && args.has_theta_matrix_type()) {
    cache_manager.RequestThetaMatrix(get_theta_matrix_args, theta_matrix);
  }
}

void MasterComponent::MergeModel(const MergeModelArgs& merge_model_args) {
  VLOG(0) << "MasterComponent: start merging models";
  if (merge_model_args.nwt_source_name_size() == 0) {
    BOOST_THROW_EXCEPTION(InvalidOperation("MergeModelArgs.nwt_source_name must not be empty"));
  }
  if (merge_model_args.nwt_source_name_size() != merge_model_args.source_weight_size()) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
        "MergeModelArgs.nwt_source_name_size() != MergeModelArgs.source_weight_size()"));
  }

  std::shared_ptr<MasterModelConfig> config = instance_->config();
  if (config != nullptr) {
    MergeModelArgs* mutable_args = const_cast<MergeModelArgs*>(&merge_model_args);
    if (merge_model_args.topic_name_size() == 0) {
      mutable_args->mutable_topic_name()->CopyFrom(config->topic_name());
    }
    FixMessage(mutable_args);
  }

  if (merge_model_args.topic_name_size() == 0) {
    // Auto-detect topic names from a source matrix (the first one in the list)
    for (int i = 0; i < merge_model_args.nwt_source_name_size(); ++i) {
      ModelName model_name = merge_model_args.nwt_source_name(i);
      std::shared_ptr<const PhiMatrix> phi_matrix = instance_->GetPhiMatrix(model_name);
      if (phi_matrix != nullptr) {
        const_cast<MergeModelArgs*>(&merge_model_args)->mutable_topic_name()->CopyFrom(phi_matrix->topic_name());
        break;
      }
    }
  }

  std::shared_ptr<DensePhiMatrix> nwt_target = std::make_shared<DensePhiMatrix>(
    merge_model_args.nwt_target_name(), merge_model_args.topic_name());

  std::shared_ptr<Dictionary> dictionary = nullptr;
  if (merge_model_args.has_dictionary_name()) {
    dictionary = instance_->dictionaries()->get(merge_model_args.dictionary_name());
    if (dictionary == nullptr || dictionary->size() == 0) {
      BOOST_THROW_EXCEPTION(InvalidOperation("Dictionary " +
        merge_model_args.dictionary_name() + " does not exist or has no tokens"));
    }

    for (int token_index = 0; token_index < (int64_t) dictionary->size(); ++token_index) {
      nwt_target->AddToken(dictionary->entry(token_index)->token());
    }
  }

  std::stringstream ss;
  for (int i = 0; i < merge_model_args.nwt_source_name_size(); ++i) {
    ModelName model_name = merge_model_args.nwt_source_name(i);
    ss << (i == 0 ? "" : ", ") << model_name;

    float weight = merge_model_args.source_weight(i);

    std::shared_ptr<const PhiMatrix> phi_matrix = instance_->GetPhiMatrix(model_name);
    if (phi_matrix == nullptr) {
      LOG(WARNING) << "Model " << model_name << " does not exist";
      continue;
    }
    const PhiMatrix& n_wt = *phi_matrix;

    if (n_wt.token_size() > 0) {
      ::artm::TopicModel topic_model;
      PhiMatrixOperations::RetrieveExternalTopicModel(n_wt, GetTopicModelArgs(), &topic_model);
      const bool add_missing_tokens = (dictionary == nullptr);
      PhiMatrixOperations::ApplyTopicModelOperation(topic_model, weight, add_missing_tokens, nwt_target.get());
    }
  }

  if (nwt_target == nullptr) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
        "ArtmMergeModel() have not found any models to merge. "
        "Verify that at least one of the following models exist: " + ss.str()));
  }
  instance_->SetPhiMatrix(merge_model_args.nwt_target_name(), nwt_target);
  VLOG(0) << "MasterComponent: complete merging models";
}

void MasterComponent::RegularizeModel(const RegularizeModelArgs& regularize_model_args) {
  VLOG(0) << "MasterComponent: start regularizing model " << regularize_model_args.pwt_source_name();
  const std::string& pwt_source_name = regularize_model_args.pwt_source_name();
  const std::string& nwt_source_name = regularize_model_args.nwt_source_name();
  const std::string& rwt_target_name = regularize_model_args.rwt_target_name();

  if (!regularize_model_args.has_pwt_source_name()) {
    BOOST_THROW_EXCEPTION(InvalidOperation("RegularizeModelArgs.pwt_source_name is missing"));
  }
  if (!regularize_model_args.has_nwt_source_name()) {
    BOOST_THROW_EXCEPTION(InvalidOperation("RegularizeModelArgs.nwt_source_name is missing"));
  }
  if (!regularize_model_args.has_rwt_target_name()) {
    BOOST_THROW_EXCEPTION(InvalidOperation("RegularizeModelArgs.rwt_target_name is missing"));
  }

  std::shared_ptr<const PhiMatrix> nwt_phi_matrix = instance_->GetPhiMatrixSafe(nwt_source_name);
  const PhiMatrix& n_wt = *nwt_phi_matrix;

  std::shared_ptr<const PhiMatrix> pwt_phi_matrix = instance_->GetPhiMatrixSafe(pwt_source_name);
  const PhiMatrix& p_wt = *pwt_phi_matrix;

  auto rwt_target(std::make_shared<DensePhiMatrix>(rwt_target_name, nwt_phi_matrix->topic_name()));
  rwt_target->Reshape(*nwt_phi_matrix);
  PhiMatrixOperations::InvokePhiRegularizers(instance_.get(), regularize_model_args.regularizer_settings(),
                                             p_wt, n_wt, rwt_target.get());
  instance_->SetPhiMatrix(rwt_target_name, rwt_target);
  VLOG(0) << "MasterComponent: complete regularizing model " << regularize_model_args.pwt_source_name();
}

void MasterComponent::NormalizeModel(const NormalizeModelArgs& normalize_model_args) {
  VLOG(0) << "MasterComponent: start normalizing model " << normalize_model_args.nwt_source_name();
  const std::string& pwt_target_name = normalize_model_args.pwt_target_name();
  const std::string& nwt_source_name = normalize_model_args.nwt_source_name();
  const std::string& rwt_source_name = normalize_model_args.rwt_source_name();

  if (!normalize_model_args.has_pwt_target_name()) {
    BOOST_THROW_EXCEPTION(InvalidOperation("NormalizeModelArgs.pwt_target_name is missing"));
  }
  if (!normalize_model_args.has_nwt_source_name()) {
    BOOST_THROW_EXCEPTION(InvalidOperation("NormalizeModelArgs.pwt_target_name is missing"));
  }

  std::shared_ptr<const PhiMatrix> nwt_phi_matrix = instance_->GetPhiMatrixSafe(nwt_source_name);
  const PhiMatrix& n_wt = *nwt_phi_matrix;

  std::shared_ptr<const PhiMatrix> rwt_phi_matrix;
  if (normalize_model_args.has_rwt_source_name()) {
    rwt_phi_matrix = instance_->GetPhiMatrixSafe(rwt_source_name);
  }

  auto pwt_target(std::make_shared<DensePhiMatrix>(pwt_target_name, n_wt.topic_name()));
  pwt_target->Reshape(n_wt);
  if (rwt_phi_matrix == nullptr) {
    PhiMatrixOperations::FindPwt(n_wt, pwt_target.get());
  } else {
    PhiMatrixOperations::FindPwt(n_wt, *rwt_phi_matrix, pwt_target.get());
  }
  instance_->SetPhiMatrix(pwt_target_name, pwt_target);
  VLOG(0) << "MasterComponent: complete normalizing model " << normalize_model_args.nwt_source_name();
}

void MasterComponent::OverwriteTopicModel(const ::artm::TopicModel& args) {
  std::shared_ptr<MasterModelConfig> config = instance_->config();
  if (config != nullptr && !args.has_name()) {
    const_cast< ::artm::TopicModel*>(&args)->set_name(config->pwt_name());
  }

  auto target = std::make_shared<DensePhiMatrix>(args.name(), args.topic_name());
  PhiMatrixOperations::ApplyTopicModelOperation(args, 1.0f, /* add_missing_tokens = */ true, target.get());
  instance_->SetPhiMatrix(args.name(), target);
}

void MasterComponent::Request(const GetThetaMatrixArgs& args, ::artm::ThetaMatrix* result) {
  instance_->cache_manager()->RequestThetaMatrix(args, result);
}

static void ValidateProcessedItems(std::string method_description, MasterComponent* master) {
  ::artm::GetScoreValueArgs get_items_processed;
  ::artm::ScoreData items_processed_data;
  get_items_processed.set_score_name("^^^ItemsProcessedScore^^^");
  master->Request(get_items_processed, &items_processed_data);
  ::artm::ItemsProcessedScore items_processed;
  items_processed.ParseFromString(items_processed_data.data());
  LOG(INFO) << method_description << ": " << DescribeMessage(items_processed);

  if (items_processed.num_batches() == 0) {
    BOOST_THROW_EXCEPTION(InvalidOperation(method_description + ": no batches to process"));
  }
  if (items_processed.value() == 0) {
    BOOST_THROW_EXCEPTION(InvalidOperation(method_description + ": no items to process --- all batches were empty"));
  }
  if (items_processed.token_weight() == 0) {
    BOOST_THROW_EXCEPTION(InvalidOperation(method_description + ": no tokens to process --- all items were empty"));
  }
  if (items_processed.token_weight_in_effect() == 0) {
    BOOST_THROW_EXCEPTION(InvalidOperation(method_description +
        ": no tokens in effect --- either tokens not present in the model, or tokens were ignored due to class_id"));
  }
}

void MasterComponent::Request(const GetThetaMatrixArgs& args,
                              ::artm::ThetaMatrix* result,
                              std::string* external) {
  Request(args, result);
  if (args.matrix_layout() == artm::MatrixLayout_Sparse) {
    HandleSparseThetaMatrixRequest(result, external);
  } else {
    HandleExternalThetaMatrixRequest(result, external);
  }
}

// ToDo(sashafrey): what should be the default cache policy for TransformMasterModel?
//                  Currently it saves the result in the cache. The result is then empty...
void MasterComponent::Request(const TransformMasterModelArgs& args, ::artm::ThetaMatrix* result) {
  std::shared_ptr<MasterModelConfig> config = instance_->config();
  if (config == nullptr) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
        "Invalid master_id; use ArtmCreateMasterModel instead of ArtmCreateMasterComponent"));
  }

  if (args.theta_matrix_type() == ThetaMatrixType_Cache) {
    ClearThetaCache(ClearThetaCacheArgs());
  }
  ClearScoreCache(ClearScoreCacheArgs());

  ProcessBatchesArgs process_batches_args;
  process_batches_args.mutable_batch_filename()->CopyFrom(args.batch_filename());
  process_batches_args.mutable_batch()->CopyFrom(args.batch());
  process_batches_args.set_pwt_source_name(config->pwt_name());
  if (config->has_num_document_passes()) {
    process_batches_args.set_num_document_passes(config->num_document_passes());
  }
  for (const auto& regularizer : config->regularizer_config()) {
    process_batches_args.add_regularizer_name(regularizer.name());
    process_batches_args.add_regularizer_tau(regularizer.tau());
  }

  if (config->has_opt_for_avx()) {
    process_batches_args.set_opt_for_avx(config->opt_for_avx());
  }
  if (config->has_reuse_theta()) {
    process_batches_args.set_reuse_theta(config->reuse_theta());
  }

  process_batches_args.mutable_class_id()->CopyFrom(config->class_id());
  process_batches_args.mutable_class_weight()->CopyFrom(config->class_weight());

  process_batches_args.mutable_transaction_typename()->CopyFrom(config->transaction_typename());
  process_batches_args.mutable_transaction_weight()->CopyFrom(config->transaction_weight());

  process_batches_args.set_theta_matrix_type(args.theta_matrix_type());
  if (args.has_predict_class_id()) {
    process_batches_args.set_predict_class_id(args.predict_class_id());
  }

  FixMessage(&process_batches_args);

  BatchManager batch_manager;
  RequestProcessBatchesImpl(process_batches_args, &batch_manager,
                            /* async =*/ false, /*score_manager =*/ nullptr, result);
  ValidateProcessedItems("Transform", this);
}

void MasterComponent::Request(const TransformMasterModelArgs& args,
                              ::artm::ThetaMatrix* result,
                              std::string* external) {
  const bool is_dense_theta = args.theta_matrix_type() == artm::ThetaMatrixType_Dense;
  const bool is_dense_ptdw = args.theta_matrix_type() == artm::ThetaMatrixType_DensePtdw;
  const bool is_sparse_theta = args.theta_matrix_type() == artm::ThetaMatrixType_Sparse;
  if (!is_dense_theta && !is_dense_ptdw && !is_sparse_theta) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
        "Dense or sparse matrix format is required for ArtmRequestProcessBatchesExternal"));
  }

  Request(args, result);
  if (is_sparse_theta) {
    HandleSparseThetaMatrixRequest(result, external);
  } else {
    HandleExternalThetaMatrixRequest(result, external);
  }
}

class BatchesIterator {
 public:
  virtual ~BatchesIterator() { }
  virtual void move(ProcessBatchesArgs* args) = 0;
};

class OfflineBatchesIterator : public BatchesIterator {
 public:
  OfflineBatchesIterator(const ::google::protobuf::RepeatedPtrField<std::string>& batch_filename,
                         const ::google::protobuf::RepeatedField<float>& batch_weight)
      : batch_filename_(batch_filename)
      , batch_weight_(batch_weight) { }

  virtual ~OfflineBatchesIterator() { }

 private:
  const ::google::protobuf::RepeatedPtrField<std::string>& batch_filename_;
  const ::google::protobuf::RepeatedField<float>& batch_weight_;

  virtual void move(ProcessBatchesArgs* args) {
    args->mutable_batch_filename()->CopyFrom(batch_filename_);
    args->mutable_batch_weight()->CopyFrom(batch_weight_);
  }
};

class OnlineBatchesIterator : public BatchesIterator {
 public:
  OnlineBatchesIterator(const ::google::protobuf::RepeatedPtrField<std::string>& batch_filename,
                        const ::google::protobuf::RepeatedField<float>& batch_weight,
                        const ::google::protobuf::RepeatedField<int>& update_after,
                        const ::google::protobuf::RepeatedField<float>& apply_weight,
                        const ::google::protobuf::RepeatedField<float>& decay_weight)
      : batch_filename_(batch_filename)
      , batch_weight_(batch_weight)
      , update_after_(update_after)
      , apply_weight_(apply_weight)
      , decay_weight_(decay_weight)
      , current_(0) { }

  virtual ~OnlineBatchesIterator() { }

  bool more() const { return current_ < static_cast<int>(update_after_.size()); }

  virtual void move(ProcessBatchesArgs* args) {
    args->clear_batch_filename();
    args->clear_batch_weight();

    if (static_cast<int>(current_) >= update_after_.size()) {
      return;
    }

    unsigned first = (current_ == 0) ? 0 : update_after_.Get(current_ - 1);
    unsigned last = update_after_.Get(current_);
    for (unsigned i = first; i < last; ++i) {
      args->add_batch_filename(batch_filename_.Get(i));
      args->add_batch_weight(batch_weight_.Get(i));
    }

    current_++;
  }

  float apply_weight() { return apply_weight_.Get(current_); }
  float decay_weight() { return decay_weight_.Get(current_); }
  int update_after() { return update_after_.Get(current_); }

  float apply_weight(int index) { return apply_weight_.Get(index); }
  float decay_weight(int index) { return decay_weight_.Get(index); }
  int update_after(int index) { return update_after_.Get(index); }

  void reset() { current_ = 0; }

 private:
  const ::google::protobuf::RepeatedPtrField<std::string>& batch_filename_;
  const ::google::protobuf::RepeatedField<float>& batch_weight_;
  const ::google::protobuf::RepeatedField<int>& update_after_;
  const ::google::protobuf::RepeatedField<float>& apply_weight_;
  const ::google::protobuf::RepeatedField<float>& decay_weight_;
  unsigned current_;  // index in update_after_ array
};

class StringIndex {
 public:
  explicit StringIndex(std::string prefix) : i_(0), prefix_(prefix) { }
  StringIndex(std::string prefix, int i) : i_(i), prefix_(prefix) { }
  int get_index() { return i_; }
  operator std::string() const { return prefix_ + boost::lexical_cast<std::string>(i_); }
  StringIndex operator+(int offset) { return StringIndex(prefix_, i_ + offset); }
  StringIndex operator-(int offset) { return StringIndex(prefix_, i_ - offset); }
  int operator++() { return ++i_; }
  int operator++(int) { return i_++; }

 private:
  int i_;
  std::string prefix_;
};

class ArtmExecutor {
 public:
  ArtmExecutor(const MasterModelConfig& master_model_config,
               MasterComponent* master_component)
      : master_model_config_(master_model_config),
        pwt_name_(master_model_config.pwt_name()),
        nwt_name_(master_model_config.nwt_name()),
        master_component_(master_component) {
    if (master_model_config.has_num_document_passes()) {
      process_batches_args_.set_num_document_passes(master_model_config.num_document_passes());
    }

    process_batches_args_.mutable_class_id()->CopyFrom(master_model_config.class_id());
    process_batches_args_.mutable_class_weight()->CopyFrom(master_model_config.class_weight());

    process_batches_args_.mutable_transaction_typename()->CopyFrom(master_model_config.transaction_typename());
    process_batches_args_.mutable_transaction_weight()->CopyFrom(master_model_config.transaction_weight());

    for (const auto& regularizer : master_model_config.regularizer_config()) {
      process_batches_args_.add_regularizer_name(regularizer.name());
      process_batches_args_.add_regularizer_tau(regularizer.tau());
    }

    for (const auto& regularizer : master_model_config.regularizer_config()) {
      RegularizerSettings* settings = regularize_model_args_.add_regularizer_settings();
      settings->set_tau(regularizer.tau());
      settings->set_name(regularizer.name());
      if (regularizer.has_gamma()) {
        settings->set_gamma(regularizer.gamma());
      }
    }

    if (master_model_config.has_opt_for_avx()) {
      process_batches_args_.set_opt_for_avx(master_model_config.opt_for_avx());
    }
    if (master_model_config.has_reuse_theta()) {
      process_batches_args_.set_reuse_theta(master_model_config.reuse_theta());
    }
  }

  void ExecuteOfflineAlgorithm(int num_collection_passes, OfflineBatchesIterator* iter) {
    const std::string rwt_name = "rwt";
    master_component_->ClearScoreCache(ClearScoreCacheArgs());
    for (int pass = 0; pass < num_collection_passes; ++pass) {
      ::artm::core::ScoreManager score_manager(master_component_->instance_.get());
      ProcessBatches(pwt_name_, nwt_name_, iter, &score_manager);
      Regularize(pwt_name_, nwt_name_, rwt_name);
      Normalize(pwt_name_, nwt_name_, rwt_name);
      StoreScores(&score_manager);
    }

    Dispose(rwt_name);
  }

  void ExecuteOnlineAlgorithm(OnlineBatchesIterator* iter) {
    const std::string rwt_name = "rwt";
    StringIndex nwt_hat_index("nwt_hat");

    master_component_->ClearScoreCache(ClearScoreCacheArgs());
    while (iter->more()) {
      float apply_weight = iter->apply_weight();
      float decay_weight = iter->decay_weight();

      ::artm::core::ScoreManager score_manager(master_component_->instance_.get());
      ProcessBatches(pwt_name_, nwt_hat_index, iter, &score_manager);
      Merge(nwt_name_, decay_weight, nwt_hat_index, apply_weight);
      Dispose(nwt_hat_index);
      Regularize(pwt_name_, nwt_name_, rwt_name);
      Normalize(pwt_name_, nwt_name_, rwt_name);
      StoreScores(&score_manager);

      nwt_hat_index++;
    }  // while (iter->more())

    iter->reset();
  }

  void ExecuteAsyncOnlineAlgorithm(OnlineBatchesIterator* iter) {
    /**************************************************
    1. Enough batches.
    i = 0: process(b1, pwt,  nwt0)
    i = 1: process(b2, pwt,  nwt1) wait(nwt0) merge(nwt, nwt0) dispose(nwt0) regularize(pwt,  nwt, rwt) normalize(nwt, rwt, pwt2) dispose(pwt0)
    i = 2: process(b3, pwt2, nwt2) wait(nwt1) merge(nwt, nwt1) dispose(nwt1) regularize(pwt2, nwt, rwt) normalize(nwt, rwt, pwt3) dispose(pwt1)
    i = 3: process(b4, pwt3, nwt3) wait(nwt2) merge(nwt, nwt2) dispose(nwt2) regularize(pwt3, nwt, rwt) normalize(nwt, rwt, pwt4) dispose(pwt2)
    i = 4: process(b5, pwt4, nwt4) wait(nwt3) merge(nwt, nwt3) dispose(nwt3) regularize(pwt4, nwt, rwt) normalize(nwt, rwt, pwt5) dispose(pwt3)
    i = 4:                         wait(nwt4) merge(nwt, nwt4) dispose(nwt4) regularize(pwt5, nwt, rwt) normalize(nwt, rwt, pwt)  dispose(pwt4) dispose(pwt5)
    2. Not enough batches -- same code works just fine.
    i = 0: process(b1, pwt,  nwt0)
    i = 1:                         wait(nwt0) merge(nwt, nwt0) dispose(nwt0) regularize(pwt,  nwt, rwt) normalize(nwt, rwt, pwt)  dispose(pwt0) dispose(pwt1)
    **************************************************/

    const std::string rwt_name = "rwt";
    std::string pwt_active = pwt_name_;
    StringIndex pwt_index("pwt");
    StringIndex nwt_hat_index("nwt_hat");

    master_component_->ClearScoreCache(ClearScoreCacheArgs());
    int op_id = AsyncProcessBatches(pwt_active, nwt_hat_index, iter);

    while (true) {
      bool is_last = !iter->more();
      pwt_index++; nwt_hat_index++;

      float apply_weight = iter->apply_weight(op_id);
      float decay_weight = iter->decay_weight(op_id);

      int temp_op_id = op_id;
      if (!is_last) {
        op_id = AsyncProcessBatches(pwt_active, nwt_hat_index, iter);
      }

      Await(temp_op_id);
      Merge(nwt_name_, decay_weight, nwt_hat_index - 1, apply_weight);
      Dispose(nwt_hat_index - 1);
      Regularize(pwt_active, nwt_name_, rwt_name);

      pwt_active = is_last ? pwt_name_ : std::string(pwt_index + 1);
      Normalize(pwt_active, nwt_name_, rwt_name);

      Dispose(pwt_index - 1);
      if (is_last) {
        Dispose(pwt_index);
      }
      if (is_last) {
        break;
      }
    }

    iter->reset();
  }

  ProcessBatchesArgs* mutable_process_batches_args() {
    return &process_batches_args_;
  }

 private:
  const MasterModelConfig& master_model_config_;
  const std::string& pwt_name_;
  const std::string& nwt_name_;

  MasterComponent* master_component_;
  ProcessBatchesArgs process_batches_args_;
  RegularizeModelArgs regularize_model_args_;
  std::vector<std::shared_ptr<BatchManager>> async_;

  void ProcessBatches(std::string pwt, std::string nwt, BatchesIterator* iter, ScoreManager* score_manager) {
    process_batches_args_.set_pwt_source_name(pwt);
    process_batches_args_.set_nwt_target_name(nwt);
    iter->move(&process_batches_args_);

    BatchManager batch_manager;
    LOG(INFO) << DescribeMessage(process_batches_args_);
    master_component_->RequestProcessBatchesImpl(process_batches_args_,
                                                 &batch_manager,
                                                 /* async =*/ false,
                                                 /* score_manager =*/ score_manager,
                                                 /* theta_matrix*/ nullptr);
    process_batches_args_.clear_batch_filename();
  }

  int AsyncProcessBatches(std::string pwt, std::string nwt, BatchesIterator* iter) {
    process_batches_args_.set_pwt_source_name(pwt);
    process_batches_args_.set_nwt_target_name(nwt);
    process_batches_args_.set_theta_matrix_type(ThetaMatrixType_None);
    iter->move(&process_batches_args_);

    int operation_id = static_cast<int>(async_.size());
    async_.push_back(std::make_shared<BatchManager>());
    LOG(INFO) << DescribeMessage(process_batches_args_);
    master_component_->RequestProcessBatchesImpl(process_batches_args_,
                                                 async_.back().get(),
                                                 /* async =*/ true,
                                                 /* score_manager =*/ nullptr,
                                                 /* theta_matrix*/ nullptr);
    process_batches_args_.clear_batch_filename();
    return operation_id;
  }

  void Await(int operation_id) {
    while (!async_[operation_id]->IsEverythingProcessed()) {
      boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));
    }
  }

  void Regularize(std::string pwt, std::string nwt, std::string rwt) {
    if (regularize_model_args_.regularizer_settings_size() > 0) {
      regularize_model_args_.set_nwt_source_name(nwt);
      regularize_model_args_.set_pwt_source_name(pwt);
      regularize_model_args_.set_rwt_target_name(rwt);
      LOG(INFO) << DescribeMessage(regularize_model_args_);
      master_component_->RegularizeModel(regularize_model_args_);
    }
  }

  void Normalize(std::string pwt, std::string nwt, std::string rwt) {
    NormalizeModelArgs normalize_model_args;
    if (regularize_model_args_.regularizer_settings_size() > 0) {
      normalize_model_args.set_rwt_source_name(rwt);
    }

    normalize_model_args.set_nwt_source_name(nwt);
    normalize_model_args.set_pwt_target_name(pwt);
    LOG(INFO) << DescribeMessage(normalize_model_args);
    master_component_->NormalizeModel(normalize_model_args);
  }

  void StoreScores(::artm::core::ScoreManager* score_manager) {
    auto config = master_component_->config();
    for (auto& score_config : config->score_config()) {
      ScoreData* score_data = master_component_->instance_->score_tracker()->Add();
      score_manager->RequestScore(score_config.name(), score_data);
    }
  }

  void Merge(std::string nwt, float decay_weight, std::string nwt_hat, float apply_weight) {
    MergeModelArgs merge_model_args;
    merge_model_args.add_nwt_source_name(nwt);
    merge_model_args.add_source_weight(decay_weight);
    merge_model_args.add_nwt_source_name(nwt_hat);
    merge_model_args.add_source_weight(apply_weight);
    merge_model_args.set_nwt_target_name(nwt);
    LOG(INFO) << DescribeMessage(merge_model_args);
    master_component_->MergeModel(merge_model_args);
  }

  void Dispose(std::string model_name) {
    LOG(INFO) << "DisposeModel " << model_name;
    master_component_->DisposeModel(model_name);
  }
};

void MasterComponent::FitOnline(const FitOnlineMasterModelArgs& args) {
  std::shared_ptr<MasterModelConfig> config = instance_->config();
  if (config == nullptr) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
        "Invalid master_id; use ArtmCreateMasterModel instead of ArtmCreateMasterComponent"));
  }

  if (config->has_parent_master_model_id()) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
      "Can not use FitOnline for hARTM, use FitOffline instead. "
      "This error happens because MasterModelConfig.parent_master_model_id is specified."));
  }

  auto pwt_matrix = instance_->GetPhiMatrix(config->pwt_name());
  auto nwt_matrix = instance_->GetPhiMatrix(config->nwt_name());
  if (pwt_matrix != nullptr && nwt_matrix != nullptr) {
    if (!PhiMatrixOperations::HasEqualShape(*pwt_matrix, *nwt_matrix)) {
      BOOST_THROW_EXCEPTION(InvalidOperation(
        "FitOnline does not support reshape of n_wt matrix. Use FitOffline instead."));
    }
  }

  ArtmExecutor artm_executor(*config, this);
  OnlineBatchesIterator iter(args.batch_filename(), args.batch_weight(), args.update_after(),
                             args.apply_weight(), args.decay_weight());
  if (args.async()) {
    artm_executor.ExecuteAsyncOnlineAlgorithm(&iter);
  } else {
    artm_executor.ExecuteOnlineAlgorithm(&iter);
  }

  ValidateProcessedItems("FitOnline", this);
}

void MasterComponent::FitOffline(const FitOfflineMasterModelArgs& args) {
  std::shared_ptr<MasterModelConfig> config = instance_->config();
  if (config == nullptr) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
        "Invalid master_id; use ArtmCreateMasterModel instead of ArtmCreateMasterComponent"));
  }

  FitOfflineMasterModelArgs* mutable_args = const_cast<FitOfflineMasterModelArgs*>(&args);
  if (args.batch_filename_size() == 0) {
    std::vector<std::string> batch_names;
    if (!args.has_batch_folder()) {
      // Default to processing all in-memory batches
      batch_names = instance_->batches()->keys();
      if (batch_names.empty()) {
        BOOST_THROW_EXCEPTION(InvalidOperation(
          "FitOfflineMasterModelArgs.batch_filename is empty. "
          "Populate this field or provide batches via ArtmImportBatches API"));
      }
    } else {
      for (const auto& batch_path : artm::core::Helpers::ListAllBatches(args.batch_folder())) {
        batch_names.push_back(batch_path.string());
      }
      if (batch_names.empty()) {
        BOOST_THROW_EXCEPTION(InvalidOperation("No batches found in " + args.batch_folder() + " folder"));
      }
    }

    for (const auto& batch_name : batch_names) {
      mutable_args->add_batch_filename(batch_name);
    }
    FixMessage(mutable_args);
  }

  std::string pseudo_batch_id;
  call_on_destruction c([&]() {  // NOLINT
    DisposeBatch(pseudo_batch_id);
  });

  if (config->has_parent_master_model_id()) {
    // Import pseudo-batch from parent master model
    ImportBatchesArgs import_batches_args;
    Batch* batch = import_batches_args.add_batch();
    batch->set_description(::artm::core::kParentPhiMatrixBatch);
    batch->set_id(boost::lexical_cast<std::string>(boost::uuids::random_generator()()));
    pseudo_batch_id = batch->id();
    ImportBatches(import_batches_args);

    // To make processing more efficient we insert pseudo-batch at the beginning of processing list,
    // in case pseudo-batch is very big. Unfortunately protobuf messages do not have
    // an operation "insert at the beginning", so we create a separate array and swap back to mutable_args.
    FitOfflineMasterModelArgs args2;
    args2.add_batch_filename(pseudo_batch_id);
    args2.add_batch_weight(config->parent_master_model_weight());
    for (int batch_index = 0; batch_index < args.batch_filename_size(); batch_index++) {
      args2.add_batch_filename(args.batch_filename(batch_index));
      args2.add_batch_weight(args.batch_weight(batch_index));
    }
    mutable_args->mutable_batch_filename()->Swap(args2.mutable_batch_filename());
    mutable_args->mutable_batch_weight()->Swap(args2.mutable_batch_weight());
  }

  ArtmExecutor artm_executor(*config, this);
  OfflineBatchesIterator iter(args.batch_filename(), args.batch_weight());
  artm_executor.mutable_process_batches_args()->set_reset_nwt(args.reset_nwt());
  artm_executor.ExecuteOfflineAlgorithm(args.num_collection_passes(), &iter);

  ValidateProcessedItems("FitOffline", this);
}

}  // namespace core
}  // namespace artm
