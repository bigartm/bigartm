// Copyright 2018, Additive Regularization of Topic Models.

#include <algorithm>

#include "artm/core/processor_helpers.h"

namespace artm {
namespace core {

void ProcessorHelpers::CreateThetaCacheEntry(ThetaMatrix* new_cache_entry_ptr,
                                             LocalThetaMatrix<float>* theta_matrix,
                                             const Batch& batch,
                                             const PhiMatrix& p_wt,
                                             const ProcessBatchesArgs& args) {
  if (new_cache_entry_ptr == nullptr) {
    return;
  }

  const int topic_size = p_wt.topic_size();
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    const Item& item = batch.item(item_index);
    new_cache_entry_ptr->add_item_id(item.id());
    new_cache_entry_ptr->add_item_title(item.has_title() ? item.title() : std::string());
    new_cache_entry_ptr->add_item_weights();
  }

  if (!args.has_predict_class_id()) {
    for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
        new_cache_entry_ptr->mutable_item_weights(item_index)->add_value((*theta_matrix)(topic_index, item_index));
      }
    }
  } else {
    const bool predict_class_id = args.has_predict_class_id();
    new_cache_entry_ptr->clear_topic_name();
    for (int token_index = 0; token_index < p_wt.token_size(); token_index++) {
      const Token& token = p_wt.token(token_index);
      if (predict_class_id && token.class_id != args.predict_class_id()) {
        continue;
      }
    }
  }
}

void ProcessorHelpers::CreatePtdwCacheEntry(ThetaMatrix* new_cache_entry_ptr,
                                            LocalPhiMatrix<float>* ptdw_matrix,
                                            const Batch& batch,
                                            int item_index,
                                            int topic_size) {
  if (new_cache_entry_ptr == nullptr) {
    return;
  }

  const Item& item = batch.item(item_index);
  for (int token_index = 0; token_index < ptdw_matrix->num_tokens(); ++token_index) {
    new_cache_entry_ptr->add_item_id(item.id());
    new_cache_entry_ptr->add_item_title(item.has_title() ? item.title() : std::string());
    auto non_zero_topic_values = new_cache_entry_ptr->add_item_weights();
    auto non_zero_topic_indices = new_cache_entry_ptr->add_topic_indices();

    for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
      float value = ptdw_matrix->operator()(token_index, topic_index);
      if (!isZero(value, kProcessorEps)) {
        // store not-null values p(t|d,w) for given d and w
        non_zero_topic_values->add_value(value);
        // store indices of these not-null values
        non_zero_topic_indices->add_value(topic_index);
      }
    }
  }
}

std::shared_ptr<LocalThetaMatrix<float>> ProcessorHelpers::InitializeTheta(int topic_size,
                                                                           const Batch& batch,
                                                                           const ProcessBatchesArgs& args,
                                                                           const ThetaMatrix* cache) {
  auto Theta = std::make_shared<LocalThetaMatrix<float>>(topic_size, batch.item_size());

  Theta->InitializeZeros();

  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    int index_of_item = -1;
    if ((cache != nullptr) && args.reuse_theta()) {
      index_of_item = repeated_field_index_of(cache->item_title(),
        batch.item(item_index).title());
    }

    if ((index_of_item != -1) && args.reuse_theta()) {
      const FloatArray& old_thetas = cache->item_weights(index_of_item);
      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
        (*Theta)(topic_index, item_index) = old_thetas.value(topic_index);
      }
    } else {
      if (args.use_random_theta()) {
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

std::shared_ptr<LocalPhiMatrix<float>>
ProcessorHelpers::InitializePhi(const Batch& batch, const ::artm::core::PhiMatrix& p_wt) {
  bool phi_is_empty = true;
  int topic_size = p_wt.topic_size();
  auto phi_matrix = std::make_shared<LocalPhiMatrix<float>>(batch.token_size(), topic_size);
  phi_matrix->InitializeZeros();
  for (int token_index = 0; token_index < batch.token_size(); ++token_index) {
    Token token = Token(batch.class_id(token_index), batch.token(token_index));

    int p_wt_token_index = p_wt.token_index(token);
    if (p_wt_token_index != ::artm::core::PhiMatrix::kUndefIndex) {
      phi_is_empty = false;
      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
        float value = p_wt.get(p_wt_token_index, topic_index);
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

  if (phi_is_empty) {
    return nullptr;
  }

  return phi_matrix;
}

void ProcessorHelpers::CreateRegularizerAgents(const Batch& batch,
                                               const ProcessBatchesArgs& args,
                                               Instance* instance,
                                               RegularizeThetaAgentCollection* theta_agents,
                                               RegularizePtdwAgentCollection* ptdw_agents) {
  for (int reg_index = 0; reg_index < args.regularizer_name_size(); ++reg_index) {
    auto& reg_name = args.regularizer_name(reg_index);
    float tau = args.regularizer_tau(reg_index);
    auto regularizer = instance->regularizers()->get(reg_name);
    if (regularizer == nullptr) {
      LOG(ERROR) << "Theta Regularizer with name <" << reg_name << "> does not exist.";
      continue;
    }

    if (theta_agents != nullptr) {
      theta_agents->AddAgent(regularizer->CreateRegularizeThetaAgent(batch, args, tau));
    }

    if (ptdw_agents != nullptr) {
      ptdw_agents->AddAgent(regularizer->CreateRegularizePtdwAgent(batch, args, tau));
    }
  }

  if (theta_agents != nullptr) {
    theta_agents->AddAgent(std::make_shared<NormalizeThetaAgent>());
  }
}

std::shared_ptr<CsrMatrix<float>> ProcessorHelpers::InitializeSparseNdw(const Batch& batch,
                                                                        const ProcessBatchesArgs& args) {
  std::vector<float> n_dw_val;
  std::vector<int> n_dw_row_ptr;
  std::vector<int> n_dw_col_ind;

  bool use_weights = false;
  std::unordered_map<ClassId, float> class_id_to_weight;
  if (args.class_id_size() != 0) {
    use_weights = true;
    for (int i = 0; i < args.class_id_size(); ++i) {
      class_id_to_weight.emplace(args.class_id(i), args.class_weight(i));
    }
  }

  float default_tt_weight = (args.transaction_typename_size() > 0) ? 0.0f : 1.0f;
  for (int i = 0; i < args.transaction_typename_size(); ++i) {
    if (args.transaction_typename(i) == DefaultTransactionTypeName) {
      default_tt_weight = args.transaction_weight(i);
    }
  }

  // For sparse case
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
    const Item& item = batch.item(item_index);

    for (int token_index = 0; token_index < item.token_id_size(); ++token_index) {
      int token_id = item.token_id(token_index);

      float class_weight = 1.0f;
      if (use_weights) {
        ClassId class_id = batch.class_id(token_id);
        auto iter = class_id_to_weight.find(class_id);
        class_weight = (iter == class_id_to_weight.end()) ? 0.0f : iter->second;
      }

      const float token_weight = item.token_weight(token_index);
      n_dw_val.push_back(default_tt_weight * class_weight * token_weight);
      n_dw_col_ind.push_back(token_id);
    }
  }

  n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
  return std::make_shared<CsrMatrix<float>>(batch.token_size(), &n_dw_val, &n_dw_row_ptr, &n_dw_col_ind);
}

void
ProcessorHelpers::FindBatchTokenIds(const Batch& batch, const PhiMatrix& phi_matrix, std::vector<int>* token_id) {
  token_id->resize(batch.token_size(), -1);
  for (int token_index = 0; token_index < batch.token_size(); ++token_index) {
    token_id->at(token_index) = phi_matrix.token_index(Token(batch.class_id(token_index), batch.token(token_index)));
  }
}

std::shared_ptr<Score> ProcessorHelpers::CalcScores(ScoreCalculatorInterface* score_calc,
                                                    const Batch& batch,
                                                    const PhiMatrix& p_wt,
                                                    const ProcessBatchesArgs& args,
                                                    const LocalThetaMatrix<float>& theta_matrix) {
  if (!score_calc->is_cumulative()) {
    return nullptr;
  }

  std::vector<Token> batch_token_dict;
  for (int token_index = 0; token_index < batch.token_size(); ++token_index) {
    batch_token_dict.push_back(Token(batch.class_id(token_index), batch.token(token_index)));
  }

  std::shared_ptr<Score> score = score_calc->CreateScore();
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    const Item& item = batch.item(item_index);

    std::vector<float> theta_vec;
    assert(theta_matrix.num_topics() == p_wt.topic_size());
    for (int topic_index = 0; topic_index < theta_matrix.num_topics(); ++topic_index) {
      theta_vec.push_back(theta_matrix(topic_index, item_index));
    }

    score_calc->AppendScore(item, batch, batch_token_dict, p_wt, args, theta_vec, score.get());
  }

  score_calc->AppendScore(batch, p_wt, args, score.get());

  return score;
}

void ProcessorHelpers::InferPtdwAndUpdateNwtSparse(const ProcessBatchesArgs& args,
                                                   const Batch& batch,
                                                   float batch_weight,
                                                   const CsrMatrix<float>& sparse_ndw,
                                                   const ::artm::core::PhiMatrix& p_wt,
                                                   const RegularizeThetaAgentCollection& theta_agents,
                                                   const RegularizePtdwAgentCollection& ptdw_agents,
                                                   LocalThetaMatrix<float>* theta_matrix,
                                                   NwtWriteAdapter* nwt_writer, util::Blas* blas,
                                                   ThetaMatrix* new_cache_entry_ptr,
                                                   ThetaMatrix* new_ptdw_cache_entry_ptr) {
  LocalThetaMatrix<float> n_td(theta_matrix->num_topics(), theta_matrix->num_items());
  LocalThetaMatrix<float> r_td(theta_matrix->num_topics(), 1);

  const int num_topics = p_wt.topic_size();
  const int docs_count = theta_matrix->num_items();

  std::vector<int> token_id, token_nwt_id;
  ProcessorHelpers::FindBatchTokenIds(batch, p_wt, &token_id);
  if (nwt_writer != nullptr) {
    ProcessorHelpers::FindBatchTokenIds(batch, *nwt_writer->n_wt(), &token_nwt_id);
  }

  for (int d = 0; d < docs_count; ++d) {
    float* ntd_ptr = &n_td(0, d);
    float* theta_ptr = &(*theta_matrix)(0, d);  // NOLINT

    const int begin_index = sparse_ndw.row_ptr()[d];
    const int end_index = sparse_ndw.row_ptr()[d + 1];
    const int local_token_size = end_index - begin_index;
    LocalPhiMatrix<float> local_phi(local_token_size, num_topics);
    LocalPhiMatrix<float> local_ptdw(local_token_size, num_topics);
    local_phi.InitializeZeros();
    bool item_has_tokens = false;
    for (int i = begin_index; i < end_index; ++i) {
      int w = sparse_ndw.col_ind()[i];
      if (token_id[w] == ::artm::core::PhiMatrix::kUndefIndex) {
        continue;
      }
      item_has_tokens = true;
      float* local_phi_ptr = &local_phi(i - begin_index, 0);
      for (int k = 0; k < num_topics; ++k) {
        local_phi_ptr[k] = p_wt.get(token_id[w], k);
      }
    }

    if (!item_has_tokens) {
      continue;  // continue to the next item
    }

    for (int inner_iter = 0; inner_iter <= args.num_document_passes(); ++inner_iter) {
      const bool last_iteration = (inner_iter == args.num_document_passes());
      for (int i = begin_index; i < end_index; ++i) {
        const float* phi_ptr = &local_phi(i - begin_index, 0);
        float* ptdw_ptr = &local_ptdw(i - begin_index, 0);

        float p_dw_val = 0.0f;
        for (int k = 0; k < num_topics; ++k) {
          float p_tdw_val = phi_ptr[k] * theta_ptr[k];
          ptdw_ptr[k] = p_tdw_val;
          p_dw_val += p_tdw_val;
        }

        if (isZero(p_dw_val)) {
          continue;
        }
        const float Z = 1.0f / p_dw_val;
        for (int k = 0; k < num_topics; ++k) {
          ptdw_ptr[k] *= Z;
        }
      }

      ptdw_agents.Apply(d, inner_iter, &local_ptdw);

      if (!last_iteration) {  // update theta matrix (except for the last iteration)
        for (int k = 0; k < num_topics; ++k) {
          ntd_ptr[k] = 0.0f;
        }
        for (int i = begin_index; i < end_index; ++i) {
          int w = sparse_ndw.col_ind()[i];
          if (token_id[w] == -1) {
            continue;
          }

          const float n_dw = sparse_ndw.val()[i];
          const float* ptdw_ptr = &local_ptdw(i - begin_index, 0);
          for (int k = 0; k < num_topics; ++k) {
            ntd_ptr[k] += n_dw * ptdw_ptr[k];
          }
        }

        for (int k = 0; k < num_topics; ++k) {
          theta_ptr[k] = ntd_ptr[k];
        }

        r_td.InitializeZeros();
        theta_agents.Apply(d, inner_iter, num_topics, theta_ptr, r_td.get_data());
      } else {  // update n_wt matrix (on the last iteration)
        if (nwt_writer != nullptr) {
          std::vector<float> values(num_topics, 0.0f);
          for (int i = begin_index; i < end_index; ++i) {
            int w = sparse_ndw.col_ind()[i];
            if (token_nwt_id[w] == -1) {
              continue;
            }

            const float n_dw = batch_weight * sparse_ndw.val()[i];
            const float* ptdw_ptr = (token_id[w] != -1) ? &local_ptdw(i - begin_index, 0) : theta_ptr;

            for (int k = 0; k < num_topics; ++k) {
              values[k] = ptdw_ptr[k] * n_dw;
            }

            nwt_writer->Store(token_nwt_id[w], values);
          }
        }
      }
    }
    CreatePtdwCacheEntry(new_ptdw_cache_entry_ptr, &local_ptdw, batch, d, num_topics);
  }
  CreateThetaCacheEntry(new_cache_entry_ptr, theta_matrix, batch, p_wt, args);
}

void ProcessorHelpers::InferThetaAndUpdateNwtSparse(const ProcessBatchesArgs& args,
                                                    const Batch& batch,
                                                    float batch_weight,
                                                    const CsrMatrix<float>& sparse_ndw,
                                                    const ::artm::core::PhiMatrix& p_wt,
                                                    const RegularizeThetaAgentCollection& theta_agents,
                                                    LocalThetaMatrix<float>* theta_matrix,
                                                    NwtWriteAdapter* nwt_writer,
                                                    util::Blas* blas,
                                                    ThetaMatrix* new_cache_entry_ptr) {
  LocalThetaMatrix<float> n_td(theta_matrix->num_topics(), theta_matrix->num_items());
  const int num_topics = p_wt.topic_size();
  const int docs_count = theta_matrix->num_items();
  const int tokens_count = batch.token_size();

  std::vector<int> token_id;
  ProcessorHelpers::FindBatchTokenIds(batch, p_wt, &token_id);

  if (args.opt_for_avx()) {
    // This version is about 40% faster than the second alternative below.
    // Both versions return 100% equal results.
    // Speedup is due to several factors:
    // 1. explicit loops instead of blas->saxpy and blas->sdot
    //    makes compiler generate AVX instructions (vectorized 128-bit float-point operations)
    // 2. better memory usage (reduced bandwith to DRAM and more sequential accesss)

    int max_local_token_size = 0;  // find the longest document from the batch
    for (int d = 0; d < docs_count; ++d) {
      const int begin_index = sparse_ndw.row_ptr()[d];
      const int end_index = sparse_ndw.row_ptr()[d + 1];
      const int local_token_size = end_index - begin_index;
      max_local_token_size = std::max(max_local_token_size, local_token_size);
    }

    LocalPhiMatrix<float> local_phi(max_local_token_size, num_topics);
    LocalThetaMatrix<float> r_td(num_topics, 1);
    std::vector<float> helper_vector(num_topics, 0.0f);

    for (int d = 0; d < docs_count; ++d) {
      float* ntd_ptr = &n_td(0, d);
      float* theta_ptr = &(*theta_matrix)(0, d);  // NOLINT

      const int begin_index = sparse_ndw.row_ptr()[d];
      const int end_index = sparse_ndw.row_ptr()[d + 1];
      local_phi.InitializeZeros();
      bool item_has_tokens = false;
      for (int i = begin_index; i < end_index; ++i) {
        int w = sparse_ndw.col_ind()[i];
        if (token_id[w] == ::artm::core::PhiMatrix::kUndefIndex) {
          continue;
        }
        item_has_tokens = true;
        float* local_phi_ptr = &local_phi(i - begin_index, 0);
        p_wt.get(token_id[w], &helper_vector);
        for (int k = 0; k < num_topics; ++k) {
          local_phi_ptr[k] = helper_vector[k];
        }
      }

      if (!item_has_tokens) {
        continue;  // continue to the next item
      }

      for (int inner_iter = 0; inner_iter < args.num_document_passes(); ++inner_iter) {
        for (int k = 0; k < num_topics; ++k) {
          ntd_ptr[k] = 0.0f;
        }

        for (int i = begin_index; i < end_index; ++i) {
          const float* phi_ptr = &local_phi(i - begin_index, 0);

          float p_dw_val = 0.0f;
          for (int k = 0; k < num_topics; ++k) {
            p_dw_val += phi_ptr[k] * theta_ptr[k];
          }
          if (isZero(p_dw_val)) {
            continue;
          }

          const float alpha = sparse_ndw.val()[i] / p_dw_val;
          for (int k = 0; k < num_topics; ++k) {
            ntd_ptr[k] += alpha * phi_ptr[k];
          }
        }

        for (int k = 0; k < num_topics; ++k) {
          theta_ptr[k] *= ntd_ptr[k];
        }

        r_td.InitializeZeros();
        theta_agents.Apply(d, inner_iter, num_topics, theta_ptr, r_td.get_data());
      }
    }
  } else {
    std::shared_ptr<LocalPhiMatrix<float>> phi_matrix_ptr = ProcessorHelpers::InitializePhi(batch, p_wt);
    if (phi_matrix_ptr == nullptr) {
      return;
    }
    const LocalPhiMatrix<float>& phi_matrix = *phi_matrix_ptr;
    for (int inner_iter = 0; inner_iter < args.num_document_passes(); ++inner_iter) {
      // helper_td will represent either n_td or r_td, depending on the context - see code below
      LocalThetaMatrix<float> helper_td(theta_matrix->num_topics(), theta_matrix->num_items());
      helper_td.InitializeZeros();

      for (int d = 0; d < docs_count; ++d) {
        for (int i = sparse_ndw.row_ptr()[d]; i < sparse_ndw.row_ptr()[d + 1]; ++i) {
          int w = sparse_ndw.col_ind()[i];
          float p_dw_val = blas->sdot(num_topics, &phi_matrix(w, 0), 1, &(*theta_matrix)(0, d), 1);  // NOLINT
          if (isZero(p_dw_val)) {
            continue;
          }
          blas->saxpy(num_topics, sparse_ndw.val()[i] / p_dw_val, &phi_matrix(w, 0), 1, &helper_td(0, d), 1);
        }
      }

      AssignDenseMatrixByProduct(*theta_matrix, helper_td, theta_matrix);

      helper_td.InitializeZeros();  // from now this represents r_td
      theta_agents.Apply(inner_iter, *theta_matrix, &helper_td);
    }
  }

  CreateThetaCacheEntry(new_cache_entry_ptr, theta_matrix, batch, p_wt, args);

  if (nwt_writer == nullptr) {
    return;
  }

  std::vector<int> token_nwt_id;
  ProcessorHelpers::FindBatchTokenIds(batch, *nwt_writer->n_wt(), &token_nwt_id);

  CsrMatrix<float> sparse_nwd(sparse_ndw);
  sparse_nwd.Transpose(blas);

  std::vector<float> p_wt_local(num_topics, 0.0f);
  std::vector<float> n_wt_local(num_topics, 0.0f);
  for (int w = 0; w < tokens_count; ++w) {
    if (token_nwt_id[w] == -1) {
      continue;
    }

    if (token_id[w] != -1) {
      p_wt.get(token_id[w], &p_wt_local);
    } else {
      p_wt_local.assign(num_topics, 1.0f);
    }

    for (int i = sparse_nwd.row_ptr()[w]; i < sparse_nwd.row_ptr()[w + 1]; ++i) {
      int d = sparse_nwd.col_ind()[i];
      float p_wd_val = blas->sdot(num_topics, &p_wt_local[0], 1, &(*theta_matrix)(0, d), 1);  // NOLINT
      if (isZero(p_wd_val)) {
        continue;
      }
      blas->saxpy(num_topics, sparse_nwd.val()[i] / p_wd_val,
        &(*theta_matrix)(0, d), 1, &n_wt_local[0], 1);  // NOLINT
    }

    std::vector<float> values(num_topics, 0.0f);
    for (int topic_index = 0; topic_index < num_topics; ++topic_index) {
      values[topic_index] = p_wt_local[topic_index] * n_wt_local[topic_index];
      n_wt_local[topic_index] = 0.0f;
    }

    for (float& value : values) {
      value *= batch_weight;
    }
    nwt_writer->Store(token_nwt_id[w], values);
  }
}

}  // namespace core
}  // namespace artm
