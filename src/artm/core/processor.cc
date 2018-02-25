// Copyright 2017, Additive Regularization of Topic Models.

#include "artm/core/processor.h"

#include <stdlib.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

#include "boost/exception/diagnostic_information.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_generators.hpp"
#include "boost/uuid/uuid_io.hpp"

#include "glog/logging.h"

#include "artm/regularizer_interface.h"
#include "artm/score_calculator_interface.h"

#include "artm/core/protobuf_helpers.h"
#include "artm/core/call_on_destruction.h"
#include "artm/core/cuckoo_watch.h"
#include "artm/core/helpers.h"
#include "artm/core/batch_manager.h"
#include "artm/core/cache_manager.h"
#include "artm/core/score_manager.h"
#include "artm/core/phi_matrix.h"
#include "artm/core/phi_matrix_operations.h"
#include "artm/core/instance.h"

#include "artm/utility/blas.h"

namespace util = artm::utility;
namespace fs = boost::filesystem;
using ::util::CsrMatrix;
using ::util::LocalThetaMatrix;
using ::util::LocalPhiMatrix;

const float kProcessorEps = 1e-16f;

namespace artm {
namespace core {

namespace {
struct IntVectorHasher {
  size_t operator()(const std::vector<int>& elems) const {
    size_t hash = 0;
    for (const int e : elems) {
      boost::hash_combine<std::string>(hash, std::to_string(e));
    }
    return hash;
  }
};

struct TokenVectorHasher {
  size_t operator()(const std::vector<Token>& elems) const {
    size_t hash = 0;
    for (const auto& e : elems) {
      boost::hash_combine<size_t>(hash, e.hash());
    }
    return hash;
  }
};

typedef std::unordered_map<ClassId, std::unordered_set<TransactionType, TransactionHasher>> ClassIdToTt;
typedef std::unordered_map<std::vector<Token>, int, TokenVectorHasher> TransactionToIndex;

struct BatchTransactionInfo {
  ClassIdToTt class_id_to_tt;
  std::unordered_map<std::vector<int>, int, IntVectorHasher> transaction_ids_to_index;
  TransactionToIndex transaction_to_index;
  std::vector<std::vector<Token>> transactions;
  std::unordered_map<Token, int, TokenHasher> token_to_index;

  BatchTransactionInfo(const ClassIdToTt& _class_id_to_tt,
                       const std::unordered_map<std::vector<int>, int, IntVectorHasher>& _transaction_ids_to_index,
                       const TransactionToIndex& _transaction_to_index,
                       const std::vector<std::vector<Token>>& _transactions,
                       const std::unordered_map<Token, int, TokenHasher>& _token_to_index)
      : class_id_to_tt(_class_id_to_tt)
      , transaction_ids_to_index(_transaction_ids_to_index)
      , transaction_to_index(_transaction_to_index)
      , transactions(_transactions)
      , token_to_index(_token_to_index){ }
};

std::shared_ptr<BatchTransactionInfo> GetBatchTransactionsInfo(const Batch& batch) {
  ClassIdToTt class_id_to_tt;
  std::unordered_map<std::vector<int>, int, IntVectorHasher> transaction_ids_to_index;
  TransactionToIndex transaction_to_index;
  std::vector<std::vector<Token>> transactions;
  std::unordered_map<Token, int, TokenHasher> token_to_index;

  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    const Item& item = batch.item(item_index);

    for (int token_index = 0; token_index < item.transaction_start_index_size(); ++token_index) {
      const int start_index = item.transaction_start_index(token_index);
      const int end_index = (token_index + 1) < item.transaction_start_index_size() ?
                            item.transaction_start_index(token_index + 1) :
                            item.transaction_token_id_size();
      std::vector<int> vec;
      for (int i = start_index; i < end_index; ++i) {
        vec.push_back(item.transaction_token_id(i));
      }
      auto iter = transaction_ids_to_index.find(vec);
      if (iter == transaction_ids_to_index.end()) {
        transaction_ids_to_index.insert(std::make_pair(vec, transaction_ids_to_index.size()));

        std::string str;
        for (int token_id = start_index; token_id < end_index; ++token_id) {
          auto& tmp = batch.class_id(item.transaction_token_id(token_id));
          str += (token_id == start_index) ? tmp : TransactionSeparator + tmp;
        }

        TransactionType tt(str);
        for (const ClassId& class_id : tt.AsVector()) {
          class_id_to_tt[class_id].emplace(tt);
        }

        std::vector<Token> transaction;
        for (int idx = start_index; idx < end_index; ++idx) {
          const int token_id = item.transaction_token_id(idx);
          auto token = Token(batch.class_id(token_id), batch.token(token_id), tt);
          transaction.push_back(token);
          token_to_index.insert(std::make_pair(token, token_to_index.size()));
        }
        transactions.push_back(transaction);
        transaction_to_index.insert(std::make_pair(transaction, transaction_to_index.size()));
      }
    }
  }

  if (transaction_to_index.size() != transactions.size()) {
    LOG(ERROR) << "Fatal error: transaction_to_index.size() [ " << transaction_to_index.size()
      << " ] != transactions.size() [ " << transactions.size();
  }

  if (transaction_ids_to_index.size() != transactions.size()) {
    LOG(ERROR) << "Fatal error: transaction_ids_to_index.size() [ " << transaction_ids_to_index.size()
               << " ] != transactions.size() [ " << transactions.size();
  }

  return std::make_shared<BatchTransactionInfo>(
    BatchTransactionInfo(class_id_to_tt, transaction_ids_to_index,
                         transaction_to_index, transactions, token_to_index));
}
}  // namespace

class RegularizeThetaAgentCollection : public RegularizeThetaAgent {
 private:
  std::vector<std::shared_ptr<RegularizeThetaAgent>> agents_;

 public:
  void AddAgent(std::shared_ptr<RegularizeThetaAgent> agent) {
    if (agent != nullptr) {
      agents_.push_back(agent);
    }
  }

  bool empty() const { return agents_.empty(); }

  virtual void Apply(int item_index, int inner_iter, int topics_size, const float* n_td, float* r_td) const {
    for (auto& agent : agents_) {
      agent->Apply(item_index, inner_iter, topics_size, n_td, r_td);
    }
  }

  virtual void Apply(int inner_iter, const LocalThetaMatrix<float>& n_td, LocalThetaMatrix<float>* r_td) const {
    for (auto& agent : agents_) {
      agent->Apply(inner_iter, n_td, r_td);
    }
  }
};

class RegularizePtdwAgentCollection : public RegularizePtdwAgent {
 private:
  std::vector<std::shared_ptr<RegularizePtdwAgent>> agents_;

 public:
  void AddAgent(std::shared_ptr<RegularizePtdwAgent> agent) {
    if (agent != nullptr) {
      agents_.push_back(agent);
    }
  }

  bool empty() const { return agents_.empty(); }

  virtual void Apply(int item_index, int inner_iter, LocalPhiMatrix<float>* ptdw) const {
    for (auto& agent : agents_) {
      agent->Apply(item_index, inner_iter, ptdw);
    }
  }
};

class NormalizeThetaAgent : public RegularizeThetaAgent {
 public:
  virtual void Apply(int item_index, int inner_iter, int topics_size, const float* n_td, float* r_td) const {
    float sum = 0.0f;
    for (int topic_index = 0; topic_index < topics_size; ++topic_index) {
      float val = n_td[topic_index] + r_td[topic_index];
      if (val > 0) {
        sum += val;
      }
    }

    float sum_inv = sum > 0.0f ? (1.0f / sum) : 0.0f;
    for (int topic_index = 0; topic_index < topics_size; ++topic_index) {
      float val = sum_inv * (n_td[topic_index] + r_td[topic_index]);
      if (val < 1e-16f) {
        val = 0.0f;
      }

      // Hack-hack, write normalized values back to n_td
      const_cast<float*>(n_td)[topic_index] = val;
    }
  }
};

static void CreateThetaCacheEntry(ThetaMatrix* new_cache_entry_ptr,
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

  if (!args.has_predict_transaction_type() && !args.has_predict_class_id()) {
    for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
      for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
        new_cache_entry_ptr->mutable_item_weights(item_index)->add_value((*theta_matrix)(topic_index, item_index));
      }
    }
  } else {
    const bool predict_tt = args.has_predict_transaction_type();
    const bool predict_class_id = args.has_predict_class_id();
    new_cache_entry_ptr->clear_topic_name();
    for (int token_index = 0; token_index < p_wt.token_size(); token_index++) {
      const Token& token = p_wt.token(token_index);
      if ((predict_class_id && token.class_id != args.predict_class_id() ||
          (predict_tt && token.transaction_type != TransactionType(args.predict_transaction_type())))) {
        continue;
      }

      new_cache_entry_ptr->add_topic_name(token.keyword);
      for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
        float weight = 0.0;
        for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
          weight += (*theta_matrix)(topic_index, item_index) * p_wt.get(token_index, topic_index);
        }
        new_cache_entry_ptr->mutable_item_weights(item_index)->add_value(weight);
      }
    }
  }
}

static void CreatePtdwCacheEntry(ThetaMatrix* new_cache_entry_ptr,
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
      if (std::fabs(value) > kProcessorEps) {
        // store not-null values p(t|d,w) for given d and w
        non_zero_topic_values->add_value(value);
        // store indices of these not-null values
        non_zero_topic_indices->add_value(topic_index);
      }
    }
  }
}

class NwtWriteAdapter {
 public:
  virtual void Store(int batch_token_id, int pwt_token_id, const std::vector<float>& nwt_vector) = 0;
  virtual ~NwtWriteAdapter() { }
};

class PhiMatrixWriter : public NwtWriteAdapter {
 public:
  explicit PhiMatrixWriter(PhiMatrix* n_wt) : n_wt_(n_wt) { }

  virtual void Store(int batch_token_id, int pwt_token_id, const std::vector<float>& nwt_vector) {
    assert(nwt_vector.size() == n_wt_->topic_size());
    n_wt_->increase(pwt_token_id, nwt_vector);
  }

 private:
  PhiMatrix* n_wt_;
};

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

static std::shared_ptr<LocalThetaMatrix<float>>
InitializeTheta(int topic_size, const Batch& batch, const ProcessBatchesArgs& args, const ThetaMatrix* cache) {
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

static std::shared_ptr<LocalPhiMatrix<float>>
InitializePhi(const Batch& batch,
              const ::artm::core::PhiMatrix& p_wt) {
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

static void
CreateRegularizerAgents(const Batch& batch, const ProcessBatchesArgs& args, Instance* instance,
                        RegularizeThetaAgentCollection* theta_agents, RegularizePtdwAgentCollection* ptdw_agents) {
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

static std::shared_ptr<CsrMatrix<float>>
InitializeSparseNdw(const Batch& batch, const ProcessBatchesArgs& args) {
  std::vector<float> n_dw_val;
  std::vector<int> n_dw_row_ptr;
  std::vector<int> n_dw_col_ind;

  bool use_weights = false;
  std::unordered_map<TransactionType, float, TransactionHasher> tt_to_weight;
  if (args.transaction_type_size() != 0) {
    use_weights = true;
    for (int i = 0; i < args.transaction_type_size(); ++i) {
      tt_to_weight.insert(std::make_pair(TransactionType(args.transaction_type(i)),
                                         args.transaction_weight(i)));
    }
  }

  // For sparse case
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
    const Item& item = batch.item(item_index);

    for (int token_index = 0; token_index < item.transaction_start_index_size(); ++token_index) {
      const int start_index = item.transaction_start_index(token_index);
      const int end_index = (token_index + 1) < item.transaction_start_index_size() ?
        item.transaction_start_index(token_index + 1) :
        item.transaction_token_id_size();

      for (int idx = start_index; idx < end_index; ++idx) {
        const int token_id = item.transaction_token_id(idx);
        float tt_weight = 1.0f;
        if (use_weights) {
          ClassId class_id = batch.class_id(token_id);
          auto iter = tt_to_weight.find(TransactionType(class_id));
          tt_weight = (iter == tt_to_weight.end()) ? 0.0f : iter->second;
        }
        const float token_weight = item.token_weight(token_index);
        n_dw_val.push_back(tt_weight * token_weight);
        n_dw_col_ind.push_back(token_id);
      }
    }
  }
  n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
  return std::make_shared<CsrMatrix<float>>(batch.token_size(), &n_dw_val, &n_dw_row_ptr, &n_dw_col_ind);
}

static void
InferThetaAndUpdateNwtSparse(const ProcessBatchesArgs& args, const Batch& batch, float batch_weight,
                             const CsrMatrix<float>& sparse_ndw,
                             const ::artm::core::PhiMatrix& p_wt,
                             const RegularizeThetaAgentCollection& theta_agents,
                             LocalThetaMatrix<float>* theta_matrix,
                             NwtWriteAdapter* nwt_writer, util::Blas* blas,
                             ThetaMatrix* new_cache_entry_ptr = nullptr) {
  LocalThetaMatrix<float> n_td(theta_matrix->num_topics(), theta_matrix->num_items());
  const int num_topics = p_wt.topic_size();
  const int docs_count = theta_matrix->num_items();
  const int tokens_count = batch.token_size();

  std::vector<int> token_id(batch.token_size(), -1);
  for (int token_index = 0; token_index < batch.token_size(); ++token_index) {
    token_id[token_index] = p_wt.token_index(Token(batch.class_id(token_index), batch.token(token_index)));
  }
  std::shared_ptr<LocalPhiMatrix<float>> phi_matrix_ptr = InitializePhi(batch, p_wt);
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
        if (p_dw_val == 0) {
          continue;
        }
        blas->saxpy(num_topics, sparse_ndw.val()[i] / p_dw_val, &phi_matrix(w, 0), 1, &helper_td(0, d), 1);
      }
    }

    AssignDenseMatrixByProduct(*theta_matrix, helper_td, theta_matrix);

    helper_td.InitializeZeros();  // from now this represents r_td
    theta_agents.Apply(inner_iter, *theta_matrix, &helper_td);
  }

  CreateThetaCacheEntry(new_cache_entry_ptr, theta_matrix, batch, p_wt, args);

  if (nwt_writer == nullptr) {
    return;
  }

  CsrMatrix<float> sparse_nwd(sparse_ndw);
  sparse_nwd.Transpose(blas);

  std::vector<float> p_wt_local(num_topics, 0.0f);
  std::vector<float> n_wt_local(num_topics, 0.0f);
  for (int w = 0; w < tokens_count; ++w) {
    if (token_id[w] == PhiMatrix::kUndefIndex) {
      continue;
    }
    p_wt.get(token_id[w], &p_wt_local);

    for (int i = sparse_nwd.row_ptr()[w]; i < sparse_nwd.row_ptr()[w + 1]; ++i) {
      int d = sparse_nwd.col_ind()[i];
      float p_wd_val = blas->sdot(num_topics, &p_wt_local[0], 1, &(*theta_matrix)(0, d), 1);  // NOLINT
      if (p_wd_val == 0) {
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
    nwt_writer->Store(w, token_id[w], values);
  }
}

static std::shared_ptr<CsrMatrix<float>> InitializeSparseNdx(
    const Batch& batch, const ProcessBatchesArgs& args,
    const ClassIdToTt& class_id_to_tt,
    const std::unordered_map<std::vector<int>, int, IntVectorHasher>& transaction_to_index) {
  std::vector<float> n_dw_val;
  std::vector<int> n_dw_row_ptr;
  std::vector<int> n_dw_col_ind;

  bool use_weights = false;
  std::unordered_map<TransactionType, float, TransactionHasher> tt_to_weight;
  if (args.transaction_type_size() != 0) {
    use_weights = true;
    for (int i = 0; i < args.transaction_type_size(); ++i) {
      tt_to_weight.insert(std::make_pair(TransactionType(args.transaction_type(i)),
                                         args.transaction_weight(i)));
    }
  }

  int max_doc_len = 0;
  std::vector<int> vec;
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
    const Item& item = batch.item(item_index);

    for (int token_index = 0; token_index < item.transaction_start_index_size(); ++token_index) {
      const int start_index = item.transaction_start_index(token_index);
      const int end_index = (token_index + 1) < item.transaction_start_index_size() ?
                            item.transaction_start_index(token_index + 1) :
                            item.transaction_token_id_size();

      float transaction_weight = 1.0f;
      if (use_weights) {
        std::string str;
        for (int token_id = start_index; token_id < end_index; ++token_id) {
          auto& tmp = batch.class_id(item.transaction_token_id(token_id));
          str += (token_id == start_index) ? tmp : TransactionSeparator + tmp;
        }
        auto iter = tt_to_weight.find(TransactionType(str));
        transaction_weight = (iter == tt_to_weight.end()) ? 0.0f : iter->second;
      }

      const float token_weight = item.token_weight(token_index);
      n_dw_val.push_back(transaction_weight * token_weight);

      vec.clear();
      for (int i = start_index; i < end_index; ++i) {
        vec.push_back(item.transaction_token_id(i));
      }
      auto iter = transaction_to_index.find(vec);
      if (iter != transaction_to_index.end()) {
        n_dw_col_ind.push_back(iter->second);
      } else {
        std::stringstream ss;
        ss << "Fatal error: transaction_to_index doesn't contain transaction from indices:";
        for (const int e : vec) {
          ss << " " << e;
        }
        ss << " read from item with index " << item_index << " from batch " << batch.id()
           << ", empty matrix will be returned for this batch.";
        LOG(ERROR) << ss.str();

        return std::make_shared<CsrMatrix<float>>(0, 0, 0);
      }
    }
  }
  n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));

  return std::make_shared<CsrMatrix<float>>(
    transaction_to_index.size(), &n_dw_val, &n_dw_row_ptr, &n_dw_col_ind);
}

static void
InferThetaAndUpdateNwtSparseNew(const ProcessBatchesArgs& args, const Batch& batch, float batch_weight,
                                const CsrMatrix<float>& sparse_ndx,
                                const TransactionToIndex& transaction_to_index,
                                const std::unordered_map<Token, int, TokenHasher>& token_to_local_index,
                                const std::vector<std::vector<Token>>& transactions,
                                const ::artm::core::PhiMatrix& p_wt,
                                const RegularizeThetaAgentCollection& theta_agents,
                                LocalThetaMatrix<float>* theta_matrix,
                                NwtWriteAdapter* nwt_writer, util::Blas* blas,
                                ThetaMatrix* new_cache_entry_ptr = nullptr) {
  LocalThetaMatrix<float> n_td(theta_matrix->num_topics(), theta_matrix->num_items());
  const int num_topics = p_wt.topic_size();
  const int docs_count = theta_matrix->num_items();

  LocalPhiMatrix<float> local_phi(token_to_local_index.size(), num_topics);
  LocalThetaMatrix<float> r_td(num_topics, 1);
  std::vector<float> helper_vector(num_topics, 0.0f);

  for (int d = 0; d < docs_count; ++d) {
    float* ntd_ptr = &n_td(0, d);
    float* theta_ptr = &(*theta_matrix)(0, d);  // NOLINT

    const int begin_index = sparse_ndx.row_ptr()[d];
    const int end_index = sparse_ndx.row_ptr()[d + 1];
    local_phi.InitializeZeros();
    bool item_has_tokens = false;
    for (int i = begin_index; i < end_index; ++i) {
      int w = sparse_ndx.col_ind()[i];
      auto& transaction = transactions[w];
      for (const auto& token : transaction) {
        if (p_wt.token_index(token) == ::artm::core::PhiMatrix::kUndefIndex) {
          continue;
        }
        auto iter = token_to_local_index.find(token);
        if (iter == token_to_local_index.end()) {
          continue;
        }
        item_has_tokens = true;
        float* local_phi_ptr = &local_phi(iter->second, 0);
        p_wt.get(p_wt.token_index(token), &helper_vector);
        for (int k = 0; k < num_topics; ++k) {
          local_phi_ptr[k] = helper_vector[k];
        }
      }
    }

    if (!item_has_tokens) {
      continue;  // continue to the next item
    }

    std::vector<float> p_xt_local(num_topics, 1.0f);
    for (int inner_iter = 0; inner_iter < args.num_document_passes(); ++inner_iter) {
      for (int k = 0; k < num_topics; ++k) {
        ntd_ptr[k] = 0.0f;
      }

      for (int i = begin_index; i < end_index; ++i) {
        int w = sparse_ndx.col_ind()[i];
        std::fill(p_xt_local.begin(), p_xt_local.end(), 1.0f);
        auto& transaction = transactions[w];
        for (const auto& token : transaction) {
          auto iter = token_to_local_index.find(token);

          if (iter == token_to_local_index.end()) {
            continue;
          }

          const float* phi_ptr = &local_phi(iter->second, 0);
          for (int k = 0; k < num_topics; ++k) {
            p_xt_local[k] *= phi_ptr[k];
          }
        }

        float p_dx_val = 0;
        for (int k = 0; k < num_topics; ++k) {
          p_dx_val += p_xt_local[k] * theta_ptr[k];
        }
        if (p_dx_val == 0) {
          continue;
        }

        const float alpha = sparse_ndx.val()[i] / p_dx_val;
        for (int k = 0; k < num_topics; ++k) {
          ntd_ptr[k] += alpha * p_xt_local[k];
        }
      }

      for (int k = 0; k < num_topics; ++k) {
        theta_ptr[k] *= ntd_ptr[k];
      }

      r_td.InitializeZeros();
      theta_agents.Apply(d, inner_iter, num_topics, theta_ptr, r_td.get_data());
    }
  }

  CreateThetaCacheEntry(new_cache_entry_ptr, theta_matrix, batch, p_wt, args);

  if (nwt_writer == nullptr) {
    return;
  }

  CsrMatrix<float> sparse_nxd(sparse_ndx);
  sparse_nxd.Transpose(blas);

  std::vector<float> values(num_topics, 0.0f);
  std::vector<float> p_xt_local(num_topics, 1.0f);
  for (const auto& transaction : transactions) {
    auto tr_iter = transaction_to_index.find(transaction);
    if (tr_iter == transaction_to_index.end()) {
      continue;
    }
    int transaction_index = tr_iter->second;

    std::fill(p_xt_local.begin(), p_xt_local.end(), 1.0f);
    for (const auto& token : transaction) {
      int phi_token_index = p_wt.token_index(token);
      if (phi_token_index == ::artm::core::PhiMatrix::kUndefIndex) {
        continue;
      }

      p_wt.get(phi_token_index, &helper_vector);
      for (int i = 0; i < num_topics; ++i) {
        p_xt_local[i] *= helper_vector[i];
      }
    }

    std::fill(helper_vector.begin(), helper_vector.end(), 0.0f);
    for (int i = sparse_nxd.row_ptr()[transaction_index]; i < sparse_nxd.row_ptr()[transaction_index + 1]; ++i) {
      int d = sparse_nxd.col_ind()[i];
      float p_xd_val = blas->sdot(num_topics, &p_xt_local[0], 1, &(*theta_matrix)(0, d), 1);  // NOLINT
      if (p_xd_val == 0) {
        continue;
      }

      blas->saxpy(num_topics, sparse_nxd.val()[i] / p_xd_val,
        &(*theta_matrix)(0, d), 1, &helper_vector[0], 1);  // NOLINT
    }

    for (const auto& token : transaction) {
      int phi_token_index = p_wt.token_index(token);
      if (phi_token_index == ::artm::core::PhiMatrix::kUndefIndex) {
        continue;
      }

      for (int topic_index = 0; topic_index < num_topics; ++topic_index) {
        values[topic_index] = p_xt_local[topic_index] * helper_vector[topic_index] * batch_weight;
      }

      nwt_writer->Store(-1, phi_token_index, values);
    }
  }
}

static void
InferPtdwAndUpdateNwtSparse(const ProcessBatchesArgs& args, const Batch& batch, float batch_weight,
                            const CsrMatrix<float>& sparse_ndw,
                            const ::artm::core::PhiMatrix& p_wt,
                            const RegularizeThetaAgentCollection& theta_agents,
                            const RegularizePtdwAgentCollection& ptdw_agents,
                            LocalThetaMatrix<float>* theta_matrix,
                            NwtWriteAdapter* nwt_writer, util::Blas* blas,
                            ThetaMatrix* new_cache_entry_ptr = nullptr,
                            ThetaMatrix* new_ptdw_cache_entry_ptr = nullptr) {
  LocalThetaMatrix<float> n_td(theta_matrix->num_topics(), theta_matrix->num_items());
  LocalThetaMatrix<float> r_td(theta_matrix->num_topics(), 1);

  const int num_topics = p_wt.topic_size();
  const int docs_count = theta_matrix->num_items();

  std::vector<int> token_id(batch.token_size(), -1);
  for (int token_index = 0; token_index < batch.token_size(); ++token_index) {
    token_id[token_index] = p_wt.token_index(Token(batch.class_id(token_index), batch.token(token_index)));
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

        if (p_dw_val == 0) {
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
            const float n_dw = batch_weight * sparse_ndw.val()[i];
            const float* ptdw_ptr = &local_ptdw(i - begin_index, 0);

            for (int k = 0; k < num_topics; ++k) {
              values[k] = ptdw_ptr[k] * n_dw;
            }

            int w = sparse_ndw.col_ind()[i];
            nwt_writer->Store(w, token_id[w], values);
          }
        }
      }
    }
    CreatePtdwCacheEntry(new_ptdw_cache_entry_ptr, &local_ptdw, batch, d, num_topics);
  }
  CreateThetaCacheEntry(new_cache_entry_ptr, theta_matrix, batch, p_wt, args);
}

static std::shared_ptr<Score>
CalcScores(ScoreCalculatorInterface* score_calc, const Batch& batch,
           const PhiMatrix& p_wt, const ProcessBatchesArgs& args, const LocalThetaMatrix<float>& theta_matrix) {
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

    score_calc->AppendScore(item, batch_token_dict, p_wt, args, theta_vec, score.get());
  }

  score_calc->AppendScore(batch, p_wt, args, score.get());

  return score;
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
        if (args.transaction_type_size() != args.transaction_weight_size()) {
          std::stringstream ss;
          ss << "model.transaction_type_size() [ " << args.transaction_type_size()
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

          if (!PhiMatrixOperations::HasEqualShape(*nwt_target, p_wt)) {
            LOG(ERROR) << "Models " << part->nwt_target_name() << " and "
                       << model_name << " have inconsistent shapes.";
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
          theta_matrix = InitializeTheta(p_wt.topic_size(), batch, args, cache.get());
        }

        if (p_wt.token_size() == 0) {
          LOG(INFO) << "Phi is empty, calculations for the model " + model_name +
            "would not be processed on this iteration";
          continue;
        }

        std::shared_ptr<NwtWriteAdapter> nwt_writer;
        if (nwt_target != nullptr) {
          nwt_writer = std::make_shared<PhiMatrixWriter>(const_cast<PhiMatrix*>(nwt_target.get()));
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
            CreateRegularizerAgents(batch, args, instance_, &theta_agents, &ptdw_agents);
          }

          bool use_real_transactions = false;
          for (const auto& tt : batch.transaction_type()) {
            // We assum here that batch is correct, e.g. it's transaction_type field
            // in case of regular model contains ALL class_ids from batch, not their subset.
            // Both parser and checker generates such batches.
            if (TransactionType(tt).AsVector().size() > 1) {
              use_real_transactions = true;
              break;
            }
          }

          if (ptdw_agents.empty() && !part->has_ptdw_cache_manager()) {
            if (args.opt_for_avx()) {
              // This version is about 40% faster than the second alternative below.
              // Both versions return 100% equal results.
              // Speedup is due to several factors:
              // 1. explicit loops instead of blas->saxpy and blas->sdot
              //    makes compiler generate AVX instructions (vectorized 128-bit float-point operations)
              // 2. better memory usage (reduced bandwith to DRAM and more sequential accesss)
              std::shared_ptr<BatchTransactionInfo> batch_info;
              {
                CuckooWatch cuckoo2("GetBatchTransactionsInfo", &cuckoo, kTimeLoggingThreshold);
                batch_info = GetBatchTransactionsInfo(batch);
              }

              std::shared_ptr<CsrMatrix<float>> sparse_ndx;
              {
                CuckooWatch cuckoo2("InitializeSparseNdx", &cuckoo, kTimeLoggingThreshold);
                sparse_ndx = InitializeSparseNdx(batch, args,
                                                 batch_info->class_id_to_tt,
                                                 batch_info->transaction_ids_to_index);
              }

              CuckooWatch cuckoo2("InferThetaAndUpdateNwtSparseNew", &cuckoo, kTimeLoggingThreshold);
              InferThetaAndUpdateNwtSparseNew(args, batch, part->batch_weight(), *sparse_ndx,
                                              batch_info->transaction_to_index, batch_info->token_to_index,
                                              batch_info->transactions, p_wt, theta_agents,
                                              theta_matrix.get(), nwt_writer.get(),
                                              blas, new_cache_entry_ptr.get());
            } else if (!use_real_transactions) {
              std::shared_ptr<CsrMatrix<float>> sparse_ndw;
              {
                CuckooWatch cuckoo2("InitializeSparseNdw", &cuckoo, kTimeLoggingThreshold);
                sparse_ndw = InitializeSparseNdw(batch, args);
              }

              CuckooWatch cuckoo2("InferThetaAndUpdateNwtSparse", &cuckoo, kTimeLoggingThreshold);
              InferThetaAndUpdateNwtSparse(args, batch, part->batch_weight(), *sparse_ndw,
                                           p_wt, theta_agents, theta_matrix.get(), nwt_writer.get(),
                                           blas, new_cache_entry_ptr.get());
            } else {
              LOG(ERROR) << "Current version of BigARTM doesn't support"
                         << " usage of opt_for_avx option with complex transactions";
            }
          } else {
            if (!use_real_transactions) {
              std::shared_ptr<CsrMatrix<float>> sparse_ndw;
              {
                CuckooWatch cuckoo2("InitializeSparseNdw", &cuckoo, kTimeLoggingThreshold);
                sparse_ndw = InitializeSparseNdw(batch, args);
              }

              CuckooWatch cuckoo2("InferPtdwAndUpdateNwtSparse", &cuckoo, kTimeLoggingThreshold);
              InferPtdwAndUpdateNwtSparse(args, batch, part->batch_weight(), *sparse_ndw,
                                          p_wt, theta_agents, ptdw_agents, theta_matrix.get(), nwt_writer.get(),
                                          blas, new_cache_entry_ptr.get(),
                                          new_ptdw_cache_entry_ptr.get());
            } else {
              LOG(ERROR) << "Current version of BigARTM doesn't support"
                         << " ptdw matrix operations with with complex transactions";
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

          auto score_value = CalcScores(score_calc.get(), batch, p_wt, args, *theta_matrix);
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
