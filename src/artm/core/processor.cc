// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/processor.h"

#include <stdlib.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/exception/diagnostic_information.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_generators.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/functional/hash.hpp"

#include "glog/logging.h"

#include "artm/regularizer_interface.h"
#include "artm/score_calculator_interface.h"

#include "artm/core/protobuf_helpers.h"
#include "artm/core/call_on_destruction.h"
#include "artm/core/helpers.h"
#include "artm/core/instance_schema.h"
#include "artm/core/merger.h"
#include "artm/core/cache_manager.h"
#include "artm/core/topic_model.h"

#include "artm/utility/blas.h"

namespace util = artm::utility;
namespace fs = boost::filesystem;
using ::util::CsrMatrix;
using ::util::DenseMatrix;

const float kProcessorEps = 1e-16;

namespace artm {
namespace core {

Processor::Processor(ThreadSafeQueue<std::shared_ptr<ProcessorInput> >*  processor_queue,
                     ThreadSafeQueue<std::shared_ptr<ModelIncrement> >* merger_queue,
                     const Merger& merger,
                     const CacheManager& cache_manager,
                     const ThreadSafeHolder<InstanceSchema>& schema)
    : processor_queue_(processor_queue),
      merger_queue_(merger_queue),
      merger_(merger),
      cache_manager_(cache_manager),
      schema_(schema),
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

static std::shared_ptr<ModelIncrement>
InitializeModelIncrement(const Batch& batch, const ModelConfig& model_config,
                         const ::artm::core::TopicModel& topic_model) {
  std::shared_ptr<ModelIncrement> model_increment = std::make_shared<ModelIncrement>();
  model_increment->add_batch_uuid(batch.id());

  int topic_size = model_config.topics_count();

  ::artm::TopicModel* topic_model_inc = model_increment->mutable_topic_model();
  topic_model_inc->set_name(model_config.name());
  topic_model_inc->mutable_topic_name()->CopyFrom(topic_model.topic_name());
  topic_model_inc->set_topics_count(topic_model.topic_size());
  for (int token_index = 0; token_index < batch.token_size(); ++token_index) {
    Token token = Token(batch.class_id(token_index), batch.token(token_index));
    topic_model_inc->add_token(token.keyword);
    topic_model_inc->add_class_id(token.class_id);
    FloatArray* counters = topic_model_inc->add_token_weights();
    IntArray* topic_indices = topic_model_inc->add_topic_index();

    if ((model_config.class_id_size() > 0) &&
        (!repeated_field_contains(model_config.class_id(), token.class_id))) {
      topic_model_inc->add_operation_type(TopicModel_OperationType_Ignore);
      continue;
    }

    if (topic_model.has_token(token)) {
      topic_model_inc->add_operation_type(TopicModel_OperationType_Increment);
    } else {
      if (model_config.use_new_tokens())
        topic_model_inc->add_operation_type(TopicModel_OperationType_Initialize);
      else
        topic_model_inc->add_operation_type(TopicModel_OperationType_Ignore);
    }
  }

  return model_increment;
}

static void PopulateDataStreams(const MasterComponentConfig& config, const Batch& batch,
                                ::artm::core::StreamMasks* pi) {
  // loop through all streams
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

static std::shared_ptr<DenseMatrix<float>>
InitializeTheta(const Batch& batch, const ModelConfig& model_config, const DataLoaderCacheEntry* cache) {
  int topic_size = model_config.topics_count();

  std::shared_ptr<DenseMatrix<float>> Theta;
  if (model_config.use_sparse_bow()) {
    Theta = std::make_shared<DenseMatrix<float>>(topic_size, batch.item_size(), false);
  } else {
    Theta = std::make_shared<DenseMatrix<float>>(topic_size, batch.item_size());
  }

  Theta->InitializeZeros();

  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    int index_of_item = -1;
    if ((cache != nullptr) && model_config.reuse_theta()) {
      index_of_item = repeated_field_index_of(cache->item_id(),
        batch.item(item_index).id());
    }

    if ((index_of_item != -1) && model_config.reuse_theta()) {
      const FloatArray& old_thetas = cache->theta(index_of_item);
      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
        (*Theta)(topic_index, item_index) = old_thetas.value(topic_index);
      }
    } else {
      if (model_config.use_random_theta()) {
        size_t seed = 0;
        boost::hash_combine(seed, std::hash<std::string>()(batch.id()));
        boost::hash_combine(seed, std::hash<int>()(item_index));
        std::vector<float> theta_values = Helpers::GenerateRandomVector(topic_size, seed);
        for (int iTopic = 0; iTopic < topic_size; ++iTopic) {
          (*Theta)(iTopic, item_index) = theta_values[iTopic];
        }
      } else {
        const float default_theta = 1.0f / topic_size;
        for (int iTopic = 0; iTopic < topic_size; ++iTopic) {
          (*Theta)(iTopic, item_index) = default_theta;
        }
      }
    }
  }

  return Theta;
}

static std::shared_ptr<DenseMatrix<float>>
InitializePhi(const Batch& batch, const ModelConfig& model_config,
              const ::artm::core::TopicModel& topic_model) {
  bool phi_is_empty = true;
  int topic_size = topic_model.topic_size();
  auto phi_matrix = std::make_shared<DenseMatrix<float>>(batch.token_size(), topic_size);
  phi_matrix->InitializeZeros();
  for (int token_index = 0; token_index < batch.token_size(); ++token_index) {
    Token token = Token(batch.class_id(token_index), batch.token(token_index));

    if (topic_model.has_token(token)) {
      phi_is_empty = false;
      auto topic_iter = topic_model.GetTopicWeightIterator(token);
      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
        float value = topic_iter[topic_index];
        if (value < kProcessorEps) {
          // Reset small values to 0.0 to avoid performance hit.
          // http://en.wikipedia.org/wiki/Denormal_number#Performance_issues
          // http://stackoverflow.com/questions/13964606/inconsistent-multiplication-performance-with-floats
          value = 0.0f;
        }
        (*phi_matrix)(token_index, topic_index) = value;
      }
    }
  }

  if (phi_is_empty)
    return nullptr;

  return phi_matrix;
}

static std::shared_ptr<RegularizeThetaAgentCollection>
CreateRegularizerAgents(const Batch& batch, const ModelConfig& model_config, const InstanceSchema& schema) {
  auto retval = std::make_shared<RegularizeThetaAgentCollection>();

  for (int reg_index = 0; reg_index < model_config.regularizer_settings_size(); ++reg_index) {
    std::string reg_name = model_config.regularizer_settings(reg_index).name();
    double tau = model_config.regularizer_settings(reg_index).tau();
    auto regularizer = schema.regularizer(reg_name);
    if (regularizer == nullptr) {
      LOG(ERROR) << "Theta Regularizer with name <" << reg_name << "> does not exist.";
      continue;
    }

    retval->AddAgent(regularizer->CreateRegularizeThetaAgent(batch, model_config, tau));
  }

  return retval;
}

static std::shared_ptr<CsrMatrix<float>>
InitializeSparseNdw(const Batch& batch, const ModelConfig& model_config) {
  std::vector<float> n_dw_val;
  std::vector<int> n_dw_row_ptr;
  std::vector<int> n_dw_col_ind;

  bool use_classes = false;
  std::map<ClassId, float> class_id_to_weight;
  if (model_config.class_id_size() != 0) {
    use_classes = true;
    for (int i = 0; i < model_config.class_id_size(); ++i) {
      class_id_to_weight.insert(std::make_pair(model_config.class_id(i), model_config.class_weight(i)));
    }
  }

  // For sparse case
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    n_dw_row_ptr.push_back(n_dw_val.size());
    auto current_item = batch.item(item_index);
    for (auto& field : current_item.field()) {
      for (int token_index = 0; token_index < field.token_id_size(); ++token_index) {
        int token_id = field.token_id(token_index);

        float class_weight = 1.0f;
        if (use_classes) {
          ClassId class_id = batch.class_id(token_id);
          auto iter = class_id_to_weight.find(class_id);
          class_weight = (iter == class_id_to_weight.end()) ? 0.0f : iter->second;
        }

        float token_count = static_cast<float>(field.token_count(token_index));
        n_dw_val.push_back(class_weight * token_count);
        n_dw_col_ind.push_back(token_id);
      }
    }
  }

  n_dw_row_ptr.push_back(n_dw_val.size());
  return std::make_shared<CsrMatrix<float>>(batch.token_size(), &n_dw_val, &n_dw_row_ptr, &n_dw_col_ind);
}

static std::shared_ptr<DenseMatrix<float>>
InitializeDenseNdw(const Batch& batch) {
  auto n_dw = std::make_shared<DenseMatrix<float>>(batch.token_size(), batch.item_size());
  n_dw->InitializeZeros();

  for (int item_index = 0; item_index < n_dw->no_columns(); ++item_index) {
    auto current_item = batch.item(item_index);
    for (auto& field : current_item.field()) {
      for (int token_index = 0; token_index < field.token_id_size(); ++token_index) {
        int token_id = field.token_id(token_index);
        int token_count = field.token_count(token_index);
        (*n_dw)(token_id, item_index) += token_count;
      }
    }
  }

  return n_dw;
}

static void
InferThetaSparse(const ModelConfig& model_config, const Batch& batch, const InstanceSchema& schema,
                 const CsrMatrix<float>& sparse_ndw, const ::artm::core::TopicModel& topic_model,
                 DenseMatrix<float>* theta_matrix, util::Blas* blas) {
  DenseMatrix<float> n_td(theta_matrix->no_rows(), theta_matrix->no_columns(), false);
  const int topics_count = model_config.topics_count();
  const int docs_count = theta_matrix->no_columns();
  const int tokens_count = batch.token_size();

  std::shared_ptr<RegularizeThetaAgentCollection> agents = CreateRegularizerAgents(batch, model_config, schema);
  agents->AddAgent(std::make_shared<NormalizeThetaAgent>());

  std::vector<int> token_id(batch.token_size(), -1);
  for (int token_index = 0; token_index < batch.token_size(); ++token_index)
    token_id[token_index] = topic_model.token_id(Token(batch.class_id(token_index), batch.token(token_index)));

  if (model_config.opt_for_avx()) {
  // This version is about 40% faster than the second alternative below.
  // Both versions return 100% equal results.
  // Speedup is due to several factors:
  // 1. explicit loops instead of blas->saxpy and blas->sdot
  //    makes compiler generate AVX instructions (vectorized 128-bit float-point operations)
  // 2. better memory usage (reduced bandwith to DRAM and more sequential accesss)
  for (int d = 0; d < docs_count; ++d) {
    float* ntd_ptr = &n_td(0, d);
    float* theta_ptr = &(*theta_matrix)(0, d);  // NOLINT

    const int begin_index = sparse_ndw.row_ptr()[d];
    const int end_index = sparse_ndw.row_ptr()[d + 1];
    const int local_token_size = end_index - begin_index;
    DenseMatrix<float> local_phi(local_token_size, topics_count);
    local_phi.InitializeZeros();
    bool item_has_tokens = false;
    for (int i = begin_index; i < end_index; ++i) {
      int w = sparse_ndw.col_ind()[i];
      if (token_id[w] == -1) continue;
      item_has_tokens = true;
      float* local_phi_ptr = &local_phi(i - begin_index, 0);
      const float* global_phi_ptr = topic_model.GetPwt(token_id[w]);
      for (int k = 0; k < topics_count; ++k) local_phi_ptr[k] = global_phi_ptr[k];
    }

    if (!item_has_tokens) continue;  // continue to the next item

    for (int inner_iter = 0; inner_iter < model_config.inner_iterations_count(); ++inner_iter) {
      for (int k = 0; k < topics_count; ++k)
        ntd_ptr[k] = 0.0f;

      for (int i = begin_index; i < end_index; ++i) {
        const float* phi_ptr = &local_phi(i - begin_index, 0);

        float p_dw_val = 0;
        for (int k = 0; k < topics_count; ++k)
          p_dw_val += phi_ptr[k] * theta_ptr[k];
        if (p_dw_val == 0) continue;

        const float alpha = sparse_ndw.val()[i] / p_dw_val;
        for (int k = 0; k < topics_count; ++k)
          ntd_ptr[k] += alpha * phi_ptr[k];
      }

      for (int k = 0; k < topics_count; ++k)
        theta_ptr[k] *= ntd_ptr[k];

      agents->Apply(d, inner_iter, model_config.topics_count(), theta_ptr);
    }
  }
  } else {
  std::shared_ptr<DenseMatrix<float>> phi_matrix_ptr = InitializePhi(batch, model_config, topic_model);
  if (phi_matrix_ptr == nullptr) return;
  const DenseMatrix<float>& phi_matrix = *phi_matrix_ptr;
  for (int inner_iter = 0; inner_iter < model_config.inner_iterations_count(); ++inner_iter) {
    DenseMatrix<float> n_td(theta_matrix->no_rows(), theta_matrix->no_columns(), false);
    n_td.InitializeZeros();

    for (int d = 0; d < docs_count; ++d) {
      for (int i = sparse_ndw.row_ptr()[d]; i < sparse_ndw.row_ptr()[d + 1]; ++i) {
        int w = sparse_ndw.col_ind()[i];
        float p_dw_val = blas->sdot(topics_count, &phi_matrix(w, 0), 1, &(*theta_matrix)(0, d), 1);  // NOLINT
        if (p_dw_val == 0) continue;
        blas->saxpy(topics_count, sparse_ndw.val()[i] / p_dw_val, &phi_matrix(w, 0), 1, &n_td(0, d), 1);
      }
    }

    AssignDenseMatrixByProduct(*theta_matrix, n_td, theta_matrix);

    for (int item_index = 0; item_index < batch.item_size(); ++item_index)
      agents->Apply(item_index, inner_iter, model_config.topics_count(), &(*theta_matrix)(0, item_index));  // NOLINT
  }
  }
}

static void
UpdateNwtSparse(const ModelConfig& model_config, const Batch& batch, const Mask* mask, const InstanceSchema& schema,
                const CsrMatrix<float>& sparse_ndw, const ::artm::core::TopicModel& topic_model,
                const DenseMatrix<float>& theta_matrix, ModelIncrement* model_increment, util::Blas* blas) {
  const int topics_count = model_config.topics_count();
  const int docs_count = theta_matrix.no_columns();
  const int tokens_count = batch.token_size();

  CsrMatrix<float> sparse_nwd(sparse_ndw);
  sparse_nwd.Transpose(blas);

  std::vector<int> token_id(batch.token_size(), -1);
  for (int token_index = 0; token_index < batch.token_size(); ++token_index)
    token_id[token_index] = topic_model.token_id(Token(batch.class_id(token_index), batch.token(token_index)));

  // n_wt should be count for items, that have corresponding true-value in stream mask
  // from batch. Or for all items, if such mask doesn't exist
  std::vector<float> p_wt(topics_count, 0.0f);
  std::vector<float> n_wt(topics_count, 0.0f);
  for (int w = 0; w < tokens_count; ++w) {
    if (token_id[w] == -1) continue;
    if (model_increment->topic_model().operation_type(w) != TopicModel_OperationType_Increment) continue;
    const float* global_phi_ptr = topic_model.GetPwt(token_id[w]);
    for (int k = 0; k < topics_count; ++k)
      p_wt[k] = global_phi_ptr[k];

    for (int i = sparse_nwd.row_ptr()[w]; i < sparse_nwd.row_ptr()[w + 1]; ++i) {
      int d = sparse_nwd.col_ind()[i];
      if ((mask != nullptr) && (mask->value(d) == false)) continue;
      float p_wd_val = blas->sdot(topics_count, &p_wt[0], 1, &theta_matrix(0, d), 1);
      if (p_wd_val == 0) continue;
      blas->saxpy(topics_count, sparse_nwd.val()[i] / p_wd_val,
        &theta_matrix(0, d), 1, &n_wt[0], 1);
    }

    FloatArray* hat_n_wt_cur = model_increment->mutable_topic_model()->mutable_token_weights(w);
    IntArray* topic_indices = model_increment->mutable_topic_model()->mutable_topic_index(w);
    assert(hat_n_wt_cur->value_size() == 0);
    std::vector<float> values(topics_count, 0.0f);
    int nnz_values = 0;
    for (int topic_index = 0; topic_index < topics_count; ++topic_index) {
      values[topic_index] = p_wt[topic_index] * n_wt[topic_index];
      n_wt[topic_index] = 0.0f;
      if (values[topic_index] >= kProcessorEps) nnz_values++;  // Find nnz_values
    }

    if (nnz_values < (topics_count / 2)) {
      // Use sparse format
      hat_n_wt_cur->mutable_value()->Reserve(nnz_values);
      topic_indices->mutable_value()->Reserve(nnz_values);
      for (int topic_index = 0; topic_index < topics_count; ++topic_index) {
        // The next line needs to be consistent with "Find nnz_values" line above
        if (values[topic_index] >= kProcessorEps) {
          hat_n_wt_cur->mutable_value()->Add(values[topic_index]);
          topic_indices->mutable_value()->Add(topic_index);
        }
      }
    } else {
      // Use dense format
      hat_n_wt_cur->mutable_value()->Reserve(topics_count);
      for (int topic_index = 0; topic_index < topics_count; ++topic_index) {
        hat_n_wt_cur->add_value(values[topic_index]);
      }
    }
  }
}

static void
InferThetaAndUpdateNwtDense(const ModelConfig& model_config, const Batch& batch, const Mask* mask,
                            const InstanceSchema& schema, const DenseMatrix<float>& dense_ndw,
                            const ::artm::core::TopicModel& topic_model, DenseMatrix<float>* theta_matrix,
                            ModelIncrement* model_increment, util::Blas* blas) {
  std::shared_ptr<DenseMatrix<float>> phi_matrix_ptr = InitializePhi(batch, model_config, topic_model);
  if (phi_matrix_ptr == nullptr) return;
  const DenseMatrix<float>& phi_matrix = *phi_matrix_ptr;
  int topics_count = model_config.topics_count();

  DenseMatrix<float> Z(phi_matrix.no_rows(), theta_matrix->no_columns());
  Z.InitializeZeros();
  for (int inner_iter = 0; inner_iter < model_config.inner_iterations_count(); ++inner_iter) {
    blas->sgemm(util::Blas::RowMajor, util::Blas::NoTrans, util::Blas::NoTrans,
      phi_matrix.no_rows(), theta_matrix->no_columns(), phi_matrix.no_columns(), 1, phi_matrix.get_data(),
      phi_matrix.no_columns(), theta_matrix->get_data(), theta_matrix->no_columns(), 0, Z.get_data(),
      theta_matrix->no_columns());

    // Z = n_dw ./ Z
    AssignDenseMatrixByDivision(dense_ndw, Z, &Z);

    // Theta_new = Theta .* (Phi' * Z) ./ repmat(n_d, nTopics, 1);
    DenseMatrix<float> prod_trans_phi_Z(phi_matrix.no_columns(), Z.no_columns());
    prod_trans_phi_Z.InitializeZeros();

    blas->sgemm(util::Blas::RowMajor, util::Blas::Trans, util::Blas::NoTrans,
      phi_matrix.no_columns(), Z.no_columns(), phi_matrix.no_rows(), 1, phi_matrix.get_data(),
      phi_matrix.no_columns(), Z.get_data(), Z.no_columns(), 0,
      prod_trans_phi_Z.get_data(), Z.no_columns());

    AssignDenseMatrixByProduct(*theta_matrix, prod_trans_phi_Z, theta_matrix);

    std::shared_ptr<RegularizeThetaAgentCollection> agents = CreateRegularizerAgents(batch, model_config, schema);
    agents->AddAgent(std::make_shared<NormalizeThetaAgent>());

    std::vector<float> theta_copy(topics_count, 0.0f);
    for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
      for (int i = 0; i < topics_count; ++i) theta_copy[i] = (*theta_matrix)(i, item_index);
      agents->Apply(item_index, inner_iter, model_config.topics_count(), &theta_copy[0]);
      for (int i = 0; i < topics_count; ++i) (*theta_matrix)(i, item_index) = theta_copy[i];
    }
  }

  blas->sgemm(util::Blas::RowMajor, util::Blas::NoTrans, util::Blas::NoTrans,
    phi_matrix.no_rows(), theta_matrix->no_columns(), phi_matrix.no_columns(), 1, phi_matrix.get_data(),
    phi_matrix.no_columns(), theta_matrix->get_data(), theta_matrix->no_columns(), 0, Z.get_data(),
    theta_matrix->no_columns());

  AssignDenseMatrixByDivision(dense_ndw, Z, &Z);

  if (model_increment == nullptr)
    return;

  auto n_wt = std::make_shared<DenseMatrix<float>>(phi_matrix.no_rows(), phi_matrix.no_columns());
  n_wt->InitializeZeros();
  if (mask != nullptr) {
    // delete columns according to bool mask
    int true_value_count = 0;
    for (int i = 0; i < mask->value_size(); ++i) {
      if (mask->value(i) == true) true_value_count++;
    }

    DenseMatrix<float> masked_Z(Z.no_rows(), true_value_count);
    DenseMatrix<float> masked_Theta(theta_matrix->no_rows(), true_value_count);
    int real_index = 0;
    for (int i = 0; i < mask->value_size(); ++i) {
      if (mask->value(i) == true) {
        for (int j = 0; j < Z.no_rows(); ++j) {
          masked_Z(j, real_index) = Z(j, i);
        }
        for (int j = 0; j < theta_matrix->no_rows(); ++j) {
          masked_Theta(j, real_index) = (*theta_matrix)(j, i);
        }
        real_index++;
      }
    }

    DenseMatrix<float> prod_Z_Theta(masked_Z.no_rows(), masked_Theta.no_rows());
    blas->sgemm(util::Blas::RowMajor, util::Blas::NoTrans, util::Blas::Trans,
      masked_Z.no_rows(), masked_Theta.no_rows(), masked_Z.no_columns(), 1,
      masked_Z.get_data(), masked_Z.no_columns(), masked_Theta.get_data(),
      masked_Theta.no_columns(), 0, prod_Z_Theta.get_data(), masked_Theta.no_rows());

    AssignDenseMatrixByProduct(prod_Z_Theta, phi_matrix, n_wt.get());
  } else {
    DenseMatrix<float> prod_Z_Theta(Z.no_rows(), theta_matrix->no_rows());
    blas->sgemm(util::Blas::RowMajor, util::Blas::NoTrans, util::Blas::Trans,
      Z.no_rows(), theta_matrix->no_rows(), Z.no_columns(), 1, Z.get_data(),
      Z.no_columns(), theta_matrix->get_data(), theta_matrix->no_columns(), 0,
      prod_Z_Theta.get_data(), theta_matrix->no_rows());

    AssignDenseMatrixByProduct(prod_Z_Theta, phi_matrix, n_wt.get());
  }

  for (int token_index = 0; token_index < n_wt->no_rows(); ++token_index) {
    if (model_increment->topic_model().operation_type(token_index) ==
      TopicModel_OperationType_Increment) {
      FloatArray* hat_n_wt_cur = model_increment->mutable_topic_model()->mutable_token_weights(token_index);
      hat_n_wt_cur->mutable_value()->Reserve(topics_count);
      assert(hat_n_wt_cur->value_size() == 0);
      for (int topic_index = 0; topic_index < topics_count; ++topic_index) {
        float value = (*n_wt)(token_index, topic_index);
        hat_n_wt_cur->add_value(value);
      }
    }
  }
}

static std::shared_ptr<Score>
CalcScores(ScoreCalculatorInterface* score_calc, const InstanceSchema& schema, const Batch& batch,
           const TopicModel& topic_model, const ModelConfig& model_config, const DenseMatrix<float>& theta_matrix,
           const StreamMasks* stream_masks) {
  if (!score_calc->is_cumulative())
    return nullptr;

  std::vector<Token> token_dict;
  for (int token_index = 0; token_index < batch.token_size(); ++token_index) {
    token_dict.push_back(Token(batch.class_id(token_index), batch.token(token_index)));
  }

  std::shared_ptr<Score> score = score_calc->CreateScore();
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    const Item& item = batch.item(item_index);

    if (stream_masks != nullptr) {
      int index_of_stream = repeated_field_index_of(stream_masks->stream_name(), score_calc->stream_name());
      if ((index_of_stream != -1) && !stream_masks->stream_mask(index_of_stream).value(item_index))
        continue;
    }

    std::vector<float> theta_vec;
    assert(theta_matrix.no_rows() == topic_model.topic_size());
    for (int topic_index = 0; topic_index < theta_matrix.no_rows(); ++topic_index) {
      theta_vec.push_back(theta_matrix(topic_index, item_index));
    }

    score_calc->AppendScore(item, token_dict, topic_model, model_config, theta_vec, score.get());
  }

  return score;
}

void Processor::FindThetaMatrix(const Batch& batch,
                                const GetThetaMatrixArgs& args, ThetaMatrix* result,
                                const GetScoreValueArgs& score_args, ScoreData* score_result) {
  util::Blas* blas = util::Blas::mkl();
  if ((blas == nullptr) || !blas->is_loaded())
    blas = util::Blas::builtin();

  std::string model_name = args.has_model_name() ? args.model_name() : score_args.model_name();
  std::shared_ptr<const TopicModel> topic_model = merger_.GetLatestTopicModel(model_name);
  if (topic_model == nullptr)
    BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException("Unable to find topic model", model_name));

  std::shared_ptr<InstanceSchema> schema = schema_.get();
  const ModelConfig& model_config = schema->model_config(model_name);

  if (model_config.class_id_size() != model_config.class_weight_size())
    BOOST_THROW_EXCEPTION(InternalError(
    "model.class_id_size() != model.class_weight_size()"));

  int topic_size = topic_model->topic_size();
  if (topic_size != model_config.topics_count())
    BOOST_THROW_EXCEPTION(InternalError(
    "Topics count mismatch between model config and physical model representation"));


  bool use_sparse_bow = model_config.use_sparse_bow();
  std::shared_ptr<CsrMatrix<float>> sparse_ndw = use_sparse_bow ? InitializeSparseNdw(batch, model_config) : nullptr;
  std::shared_ptr<DenseMatrix<float>> dense_ndw = !use_sparse_bow ? InitializeDenseNdw(batch) : nullptr;

  std::shared_ptr<DenseMatrix<float>> theta_matrix = InitializeTheta(batch, model_config, nullptr);

  if (topic_model->token_size() == 0) {
    LOG(INFO) << "Phi is empty, calculations for the model " + model_name +
      "would not be processed on this iteration";
    return;  // return from lambda; goes to next step of std::for_each
  }

  if (model_config.use_sparse_bow()) {
    InferThetaSparse(model_config, batch, *schema, *sparse_ndw, *topic_model, theta_matrix.get(), blas);
  } else {
    // We don't need 'UpdateNwt' part, but for Dense mode it is hard to split this function.
    InferThetaAndUpdateNwtDense(model_config, batch, nullptr, *schema, *dense_ndw, *topic_model,
                                theta_matrix.get(), nullptr, blas);
  }

  if (result != nullptr) {
    DataLoaderCacheEntry cache_entry;
    cache_entry.set_model_name(model_name);
    cache_entry.mutable_topic_name()->CopyFrom(topic_model->topic_name());
    for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
      const Item& item = batch.item(item_index);
      cache_entry.add_item_id(item.id());
      cache_entry.add_item_title(item.has_title() ? item.title() : std::string());
      FloatArray* item_weights = cache_entry.add_theta();
      for (int topic_index = 0; topic_index < topic_size; ++topic_index)
        item_weights->add_value((*theta_matrix)(topic_index, item_index));
    }

    BatchHelpers::PopulateThetaMatrixFromCacheEntry(cache_entry, args, result);
  }

  if (score_result != nullptr) {
    auto score_calc = schema->score_calculator(score_args.score_name());
    if (score_calc == nullptr)
      BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException("Unable to find score calculator ", score_args.score_name()));

    auto score_value = CalcScores(score_calc.get(), *schema, batch, *topic_model, model_config, *theta_matrix, nullptr);
    score_result->set_data(score_value->SerializeAsString());
    score_result->set_type(score_calc->score_type());
    score_result->set_name(score_args.score_name());
  }
}

void Processor::ThreadFunction() {
  try {
    int total_processed_batches = 0;  // counter

    Helpers::SetThreadName(-1, "Processor thread");
    LOG(INFO) << "Processor thread started";
    int pop_retries = 0;
    const int pop_retries_max = 20;

    util::Blas* blas = util::Blas::mkl();
    if ((blas == nullptr) || !blas->is_loaded()) {
      // Avoid this warning because it only applies to non-default case "use_sparse_bow==false"
      // LOG(INFO) << "Intel Math Kernel Library is not detected, "
      //    << "using built in implementation (can be slower than MKL)";
      blas = util::Blas::builtin();
    }

    for (;;) {
      if (is_stopping) {
        LOG(INFO) << "Processor thread stopped";
        LOG(INFO) << "Total number of processed batches: " << total_processed_batches;
        break;
      }

      std::shared_ptr<ProcessorInput> part;
      if (!processor_queue_->try_pop(&part)) {
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

      call_on_destruction c([&]() {
        if (part->notifiable() != nullptr) {
          part->notifiable()->Callback(part->task_id(), part->model_name());
        }
      });

      std::shared_ptr<Batch> batch_ptr;
      if (part->has_batch_filename()) {
        try {
          CuckooWatch cuckoo2("LoadMessage", &cuckoo);
          batch_ptr.reset(new Batch());
          ::artm::core::BatchHelpers::LoadMessage(part->batch_filename(), batch_ptr.get());
        } catch (std::exception& ex) {
          LOG(ERROR) << ex.what() << ", the batch will be skipped.";
          continue;
        }
      }

      const Batch& batch = batch_ptr != nullptr ? *batch_ptr : part->batch();

      if (batch.class_id_size() != batch.token_size())
        BOOST_THROW_EXCEPTION(InternalError(
            "batch.class_id_size() != batch.token_size()"));

      std::shared_ptr<InstanceSchema> schema = schema_.get();
      const MasterComponentConfig& master_config = schema->config();

      StreamMasks stream_masks;
      PopulateDataStreams(master_config, batch, &stream_masks);

      std::shared_ptr<CsrMatrix<float>> sparse_ndw;
      std::shared_ptr<DenseMatrix<float>> dense_ndw;

      const ModelName& model_name = part->model_name();
      {
        const ModelConfig& model_config = schema->model_config(model_name);

        // do not process disabled models.
        if (!model_config.enabled()) continue;

        if (model_config.class_id_size() != model_config.class_weight_size())
          BOOST_THROW_EXCEPTION(InternalError(
              "model.class_id_size() != model.class_weight_size()"));

        std::shared_ptr<const TopicModel> topic_model = merger_.GetLatestTopicModel(model_name);
        if (topic_model == nullptr) {
          LOG(ERROR) << "Model " << model_name << " does not exist.";
          continue;
        }

        assert(topic_model.get() != nullptr);

        int topic_size = topic_model->topic_size();
        if (topic_size != model_config.topics_count())
          BOOST_THROW_EXCEPTION(InternalError(
            "Topics count mismatch between model config and physical model representation"));

        if (model_config.use_sparse_bow())
          sparse_ndw = InitializeSparseNdw(batch, model_config);

        if (!model_config.use_sparse_bow() && (dense_ndw == nullptr)) {
          // dense_ndw does not depend on model_config => used the same dense_ndw for all models with !use_sparse_bow
          dense_ndw = InitializeDenseNdw(batch);
        }


        std::shared_ptr<DataLoaderCacheEntry> cache;
        boost::uuids::uuid batch_uuid = boost::lexical_cast<boost::uuids::uuid>(batch.id());
        cache = cache_manager_.FindCacheEntry(batch_uuid, model_config.name());
        std::shared_ptr<DenseMatrix<float>> theta_matrix = InitializeTheta(batch, model_config, cache.get());

        std::shared_ptr<ModelIncrement> model_increment = InitializeModelIncrement(batch, model_config, *topic_model);

        if (topic_model->token_size() == 0) {
          LOG(INFO) << "Phi is empty, calculations for the model " + model_name +
            "would not be processed on this iteration";
          merger_queue_->push(model_increment);
          continue;
        }

        // Find and save to the variable the index of model stream in the part->stream_name() list.
        int model_stream_index = repeated_field_index_of(stream_masks.stream_name(), model_config.stream_name());
        const Mask* stream_mask = (model_stream_index != -1) ? &stream_masks.stream_mask(model_stream_index) : nullptr;

        if (model_config.use_sparse_bow()) {
          InferThetaSparse(model_config, batch, *schema, *sparse_ndw, *topic_model, theta_matrix.get(), blas);
        } else {
          InferThetaAndUpdateNwtDense(model_config, batch, stream_mask, *schema, *dense_ndw, *topic_model,
                                      theta_matrix.get(), model_increment.get(), blas);
        }

        if (schema->config().cache_theta()) {
          // Update theta cache
          std::shared_ptr<DataLoaderCacheEntry> new_cache_entry_ptr(new DataLoaderCacheEntry());
          DataLoaderCacheEntry& new_cache_entry = *new_cache_entry_ptr;
          new_cache_entry.set_batch_uuid(batch.id());
          new_cache_entry.set_model_name(model_name);
          new_cache_entry.mutable_topic_name()->CopyFrom(model_increment->topic_model().topic_name());
          for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
            const Item& item = batch.item(item_index);
            new_cache_entry.add_item_id(item.id());
            new_cache_entry.add_item_title(item.has_title() ? item.title() : std::string());
            FloatArray* cached_theta = new_cache_entry.add_theta();
            for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
              cached_theta->add_value((*theta_matrix)(topic_index, item_index));
            }
          }

          if (schema->config().has_disk_cache_path()) {
            std::string disk_cache_path = schema->config().disk_cache_path();
            boost::uuids::uuid uuid = boost::uuids::random_generator()();
            fs::path file(boost::lexical_cast<std::string>(uuid) + ".cache");
            try {
              BatchHelpers::SaveMessage(file.string(), disk_cache_path, new_cache_entry);
              new_cache_entry.set_filename((fs::path(disk_cache_path) / file).string());
              new_cache_entry.clear_theta();
              new_cache_entry.clear_item_id();
            } catch (...) {
              LOG(ERROR) << "Unable to save cache entry to " << schema->config().disk_cache_path();
            }
          }

          cache_manager_.UpdateCacheEntry(new_cache_entry_ptr);
        }

        for (int score_index = 0; score_index < master_config.score_config_size(); ++score_index) {
          const ScoreName& score_name = master_config.score_config(score_index).name();

          auto score_calc = schema->score_calculator(score_name);
          if (score_calc == nullptr) {
            LOG(ERROR) << "Unable to find score calculator '" << score_name << "', referenced by "
              << "model " << model_config.name() << ".";
            continue;
          }

          auto score_value = CalcScores(score_calc.get(), *schema, batch, *topic_model, model_config,
                                        *theta_matrix, &stream_masks);
          if (score_value == nullptr)
            continue;
          model_increment->add_score_name(score_name);
          model_increment->add_score(score_value->SerializeAsString());
        }

        {
          CuckooWatch cuckoo2("await merger queue", &cuckoo);
          // Wait until merger queue has space for a new element
          int merger_queue_max_size = schema_.get()->config().merger_queue_max_size();
          for (;;) {
            if (merger_queue_->size() < merger_queue_max_size)
              break;

            boost::this_thread::sleep(boost::posix_time::milliseconds(kIdleLoopFrequency));
          }
        }

        merger_queue_->reserve();
        if (model_config.use_sparse_bow()) {
          // Keep UpdateNwtSparse as further down in the code as possible,
          // because it consumes a lot of memory to transfer increments to merger.
          UpdateNwtSparse(model_config, batch, stream_mask, *schema, *sparse_ndw, *topic_model,
            *theta_matrix, model_increment.get(), blas);
        }
        merger_queue_->release();
        merger_queue_->push(model_increment);
      }
    }
  }
  catch (...) {
    LOG(FATAL) << boost::current_exception_diagnostic_information();
  }
}

}  // namespace core
}  // namespace artm
