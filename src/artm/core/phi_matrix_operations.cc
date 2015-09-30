// Copyright 2015, Additive Regularization of Topic Models.

#include "artm/core/phi_matrix_operations.h"

#include <assert.h>

#include <algorithm>
#include <utility>
#include <string>

#include "boost/range/adaptor/map.hpp"
#include "boost/range/algorithm/copy.hpp"

#include "artm/core/protobuf_helpers.h"
#include "artm/core/helpers.h"
#include "artm/core/dense_phi_matrix.h"
#include "artm/regularizer_interface.h"

namespace artm {
namespace core {

void PhiMatrixOperations::RetrieveExternalTopicModel(const PhiMatrix& phi_matrix,
                                                     const ::artm::GetTopicModelArgs& get_model_args,
                                                     ::artm::TopicModel* topic_model) {
  if (get_model_args.request_type() == GetTopicModelArgs_RequestType_TopicNames) {
    for (auto& topic_name : phi_matrix.topic_name())
      topic_model->add_topic_name(topic_name);
    topic_model->set_topics_count(phi_matrix.topic_size());
    return;
  }

  if (get_model_args.request_type() == GetTopicModelArgs_RequestType_Tokens) {
    for (int token_index = 0; token_index < phi_matrix.token_size(); token_index++) {
      const Token& current_token = phi_matrix.token(token_index);
      topic_model->add_token(current_token.keyword);
      topic_model->add_class_id(current_token.class_id);
    }
    return;
  }

  const bool has_sparse_format = (get_model_args.matrix_layout() == GetTopicModelArgs_MatrixLayout_Sparse);

  std::vector<int> tokens_to_use;
  if (get_model_args.token_size() > 0) {
    bool use_default_class = (get_model_args.class_id_size() == 0);

    if (!use_default_class && (get_model_args.token_size() != get_model_args.class_id_size()))
      BOOST_THROW_EXCEPTION(artm::core::InvalidOperation(
        "GetTopicModelArgs: token_size != class_id_size, both greater then zero"));

    for (int i = 0; i < get_model_args.token_size(); ++i) {
      Token token(use_default_class ? DefaultClass : get_model_args.class_id(i),
                  get_model_args.token(i));
      int token_id = phi_matrix.token_index(token);
      if (token_id != -1) {
        assert(token_id >= 0 && token_id < phi_matrix.token_size());
        tokens_to_use.push_back(token_id);
      }
    }
  } else {
    if (get_model_args.class_id_size() > 0) {
      // use all tokens from the specific classes
      for (int i = 0; i < phi_matrix.token_size(); ++i) {
        if (repeated_field_contains(get_model_args.class_id(), phi_matrix.token(i).class_id)) {
          tokens_to_use.push_back(i);
        }
      }
    } else {
      tokens_to_use.reserve(phi_matrix.token_size());
      for (int i = 0; i < phi_matrix.token_size(); ++i) {
        tokens_to_use.push_back(i);
      }
    }
  }

  std::vector<int> topics_to_use;
  if (get_model_args.topic_name_size() != 0) {
    auto this_topic_name = phi_matrix.topic_name();
    for (int i = 0; i < get_model_args.topic_name_size(); ++i) {
      int topic_index = repeated_field_index_of(this_topic_name, get_model_args.topic_name(i));
      if (topic_index == -1) {
        std::stringstream ss;
        ss << "GetTopicModelArgs.topic_name[" << i << "] == " << get_model_args.topic_name(i)
           << " does not exist in ModelConfig.topic_name";
        BOOST_THROW_EXCEPTION(artm::core::InvalidOperation(ss.str()));
      }

      assert(topic_index >= 0 && topic_index < phi_matrix.topic_size());
      topics_to_use.push_back(topic_index);
    }
  } else {
    for (int i = 0; i < phi_matrix.topic_size(); ++i)
      topics_to_use.push_back(i);
  }

  LOG(INFO) << "RetrieveExternalTopicModel() with "
            << topics_to_use.size() << " topics, "
            << tokens_to_use.size() << " tokens";

  auto this_topic_names = phi_matrix.topic_name();

  // Populate topics_count and topic_name fields in the resulting message
  for (int topic_index : topics_to_use)
    topic_model->add_topic_name(this_topic_names.Get(topic_index));
  topic_model->set_topics_count(topics_to_use.size());

  // Populate all non-internal part of the resulting message
  topic_model->set_name(phi_matrix.model_name());

  for (int token_index : tokens_to_use) {
    const Token& current_token = phi_matrix.token(token_index);
    topic_model->add_token(current_token.keyword);
    topic_model->add_class_id(current_token.class_id);
    topic_model->add_operation_type(TopicModel_OperationType_Increment);

    ::artm::FloatArray *target = topic_model->add_token_weights();

    if (!has_sparse_format) {
      target->mutable_value()->Reserve(topics_to_use.size());
      for (int topic_index : topics_to_use)
        target->add_value(phi_matrix.get(token_index, topic_index));
    } else {
      ::artm::IntArray* sparse_topic_index = topic_model->add_topic_index();
      for (int topics_to_use_index = 0; topics_to_use_index < topics_to_use.size(); topics_to_use_index++) {
        int topic_index = topics_to_use[topics_to_use_index];
        float value = phi_matrix.get(token_index, topic_index);
        if (fabs(value) > get_model_args.eps()) {
          sparse_topic_index->add_value(topics_to_use_index);
          target->add_value(value);
        }
      }
    }
  }
}

void PhiMatrixOperations::ApplyTopicModelOperation(const ::artm::TopicModel& topic_model,
                                                   float apply_weight, PhiMatrix* phi_matrix) {
  if (!Helpers::Validate(topic_model, /* throw_error=*/ false)) return;

  const bool has_sparse_format = (topic_model.topic_index_size() > 0);
  const int this_topic_size = phi_matrix->topic_size();
  std::vector<int> target_topic_index;
  if (topic_model.topic_name_size() > 0) {
    bool ok = false;
    for (auto& topic_name : topic_model.topic_name()) {
      int index = repeated_field_index_of(phi_matrix->topic_name(), topic_name);
      target_topic_index.push_back(index);
      if (index != -1) ok = true;
    }
    if (!ok) {
      LOG(ERROR) << "None of TopicModel.topic_name match topic names in target model";
      return;
    }
  } else {
    if (phi_matrix->topic_size() != topic_model.topics_count())
      BOOST_THROW_EXCEPTION(InvalidOperation("Mismatch between target topics_count and TopicModel.topics_count"));
    for (int i = 0; i < topic_model.topics_count(); ++i)
      target_topic_index.push_back(i);
  }

  bool optimized_execution = false;
  if ((apply_weight == 1.0f) && (target_topic_index.size() == this_topic_size)) {
    bool ok = true;
    for (int topic_index = 0; topic_index < target_topic_index.size(); ++topic_index) {
      if (target_topic_index[topic_index] != topic_index)
        ok = false;
    }
    optimized_execution = ok;
  }

  std::vector<Token> remove_tokens;
  for (int token_index = 0; token_index < topic_model.token_size(); ++token_index) {
    const std::string& token_keyword = topic_model.token(token_index);
    const ClassId& class_id = topic_model.class_id(token_index);
    Token token(class_id, token_keyword);
    const FloatArray& counters = topic_model.token_weights(token_index);
    const IntArray* sparse_topic_index = has_sparse_format ? &topic_model.topic_index(token_index) : nullptr;
    const bool has_sparse_format_local = (sparse_topic_index != nullptr) && (sparse_topic_index->value_size() > 0);

    TopicModel_OperationType operation_type = topic_model.operation_type(token_index);
    int current_token_id = phi_matrix->token_index(token);

    float* target;
    switch (operation_type) {
    case TopicModel_OperationType_Initialize:
      // Add new tokens discovered by processor
      if (current_token_id == -1) {
        current_token_id = phi_matrix->AddToken(token);
        std::vector<float> vec = Helpers::GenerateRandomVector(phi_matrix->topic_size(), token);
        phi_matrix->increase(current_token_id, vec);
      }
      break;

    case TopicModel_OperationType_Increment:
      if (current_token_id == -1)
        current_token_id = phi_matrix->AddToken(token);

      if (optimized_execution && !has_sparse_format_local && (counters.value_size() == this_topic_size)) {
        for (int topic_index = 0; topic_index < this_topic_size; ++topic_index)
          phi_matrix->increase(current_token_id, topic_index, counters.value(topic_index));
        break;
      }

      for (int i = 0; i < counters.value_size(); ++i) {
        int topic_index = has_sparse_format_local ? sparse_topic_index->value(i) : i;
        assert(topic_index < target_topic_index.size());
        if (target_topic_index[topic_index] == -1)
          continue;
        phi_matrix->increase(current_token_id, target_topic_index[topic_index], apply_weight * counters.value(i));
      }
      break;

    case TopicModel_OperationType_Overwrite:
      if (current_token_id == -1)
        current_token_id = phi_matrix->AddToken(token);
      for (int i = 0; i < counters.value_size(); ++i) {
        int topic_index = has_sparse_format_local ? sparse_topic_index->value(i) : i;
        assert(topic_index < target_topic_index.size());
        if (target_topic_index[topic_index] == -1)
          continue;
        phi_matrix->set(current_token_id, target_topic_index[topic_index], counters.value(i));
      }
      break;

    case TopicModel_OperationType_Remove:
      if (current_token_id != -1)
        remove_tokens.push_back(token);
      break;

    case TopicModel_OperationType_Ignore:
      // ignore token == do nothing
      break;

    default:
      BOOST_THROW_EXCEPTION(ArgumentOutOfRangeException(
        "ModelIncrement.operation_type", operation_type));
    }
  }

  if (remove_tokens.size() > 0)
    phi_matrix->RemoveTokens(remove_tokens);
}

void PhiMatrixOperations::InvokePhiRegularizers(
    std::shared_ptr<InstanceSchema> schema,
    const ::google::protobuf::RepeatedPtrField<RegularizerSettings>& regularizer_settings,
    const PhiMatrix& p_wt, const PhiMatrix& n_wt, PhiMatrix* r_wt) {

  int topic_size = n_wt.topic_size();
  int token_size = n_wt.token_size();

  DensePhiMatrix local_r_wt(ModelName(), n_wt.topic_name());
  local_r_wt.Reshape(n_wt);

  auto n_t_all = PhiMatrixOperations::FindNormalizers(n_wt);

  for (auto reg_iterator = regularizer_settings.begin();
       reg_iterator != regularizer_settings.end();
       reg_iterator++) {
    auto regularizer = schema->regularizer(reg_iterator->name().c_str());

    if (regularizer != nullptr) {
      double tau = reg_iterator->tau();
      bool relative_reg = reg_iterator->use_relative_regularization();

      if (p_wt.token_size() != n_wt.token_size() || p_wt.topic_size() != n_wt.topic_size() ||
          local_r_wt.token_size() != n_wt.token_size() || local_r_wt.topic_size() != n_wt.topic_size()) {
        LOG(ERROR) << "Inconsistent matrix size: Pwt( "
          << p_wt.token_size() << ", " << p_wt.topic_size() << ") vs Nwt("
          << n_wt.token_size() << ", " << n_wt.topic_size() << ") vs Rwt("
          << local_r_wt.token_size() << ", " << local_r_wt.topic_size() << ")";
        continue;
      }

      bool retval = regularizer->RegularizePhi(p_wt, n_wt, &local_r_wt);

      // count n and r_i for relative regularization, if necessary
      // prepare next structure with parameters:
      // pair of pairs, first pair --- n and n_t, second one --- r_i and r_it
      std::unordered_map<core::ClassId, std::pair<std::pair<double, std::vector<float> >,
        std::pair<double, std::vector<float> > > > parameters;
      std::vector<bool> topics_to_regularize;

      if (relative_reg) {
        std::vector<core::ClassId> class_ids;
        if (regularizer->class_ids_to_regularize().size() > 0) {
          auto class_ids_to_regularize = regularizer->class_ids_to_regularize();
          for (auto class_id : class_ids_to_regularize) class_ids.push_back(class_id);
        } else {
          boost::copy(n_t_all | boost::adaptors::map_keys, std::back_inserter(class_ids));
        }

        if (regularizer->topics_to_regularize().size() > 0)
          topics_to_regularize = core::is_member(n_wt.topic_name(), regularizer->topics_to_regularize());
        else
          topics_to_regularize.assign(topic_size, true);

        for (auto class_id : class_ids) {
          auto iter = n_t_all.find(class_id);
          if (iter != n_t_all.end()) {
            double n = 0.0;
            double r_i = 0.0;
            std::vector<float> r_it;
            std::vector<float> n_t = iter->second;

            for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
              if (!topics_to_regularize[topic_id]) {
                r_it.push_back(-1.0f);
                continue;
              }
              n += n_t[topic_id];

              float r_it_current = 0.0f;
              for (int token_id = 0; token_id < token_size; ++token_id) {
                if (n_wt.token(token_id).class_id != iter->first) continue;

                r_it_current += local_r_wt.get(token_id, topic_id);
              }

              r_it.push_back(r_it_current);
              r_i += r_it_current;
            }

            auto pair_n = std::pair<double, std::vector<float> >(n, n_t);
            auto pair_r = std::pair<double, std::vector<float> >(r_i, r_it);
            auto pair_data = std::pair<std::pair<double, std::vector<float> >,
              std::pair<double, std::vector<float> > >(pair_n, pair_r);
            auto pair_last = std::pair<core::ClassId,
              std::pair<std::pair<double, std::vector<float> >,
              std::pair<double, std::vector<float> > > >(iter->first, pair_data);
            parameters.insert(pair_last);
          }
        }
      }

      for (int token_id = 0; token_id < token_size; ++token_id) {
        auto iter = parameters.find(n_wt.token(token_id).class_id);
        if (relative_reg) {
          if (iter == parameters.end()) continue;
        }
        for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
          float coefficient = 1.0f;
          if (relative_reg) {
            if (!topics_to_regularize[topic_id]) continue;

            double gamma = reg_iterator->gamma();
            float n_t = iter->second.first.second[topic_id];
            float n = iter->second.first.first;
            float r_it = iter->second.second.second[topic_id];
            float r_i = iter->second.second.first;
            coefficient = static_cast<float>(gamma)* (n_t / r_it) * static_cast<float>(1 - gamma) * (n / r_i);
          }
          // update global r_wt using coefficient and tau
          float increment = coefficient * tau * local_r_wt.get(token_id, topic_id);
          r_wt->increase(token_id, topic_id, increment);
        }
      }
      local_r_wt.Reset();

      if (!retval) {
        LOG(ERROR) << "Problems with type or number of parameters in Phi regularizer <" <<
          reg_iterator->name().c_str() <<
          ">. On this iteration this regularizer was turned off.\n";
      }
    } else {
      LOG(ERROR) << "Phi Regularizer with name <" <<
        reg_iterator->name().c_str() << "> does not exist.\n";
    }
  }
}

static std::map<ClassId, std::vector<float> > FindNormalizersImpl(const PhiMatrix& n_wt, const PhiMatrix* r_wt) {
  std::map<ClassId, std::vector<float> > retval;
  assert((r_wt == nullptr) || (r_wt->token_size() == n_wt.token_size() && r_wt->topic_size() == n_wt.topic_size()));

  for (int token_id = 0; token_id < n_wt.token_size(); ++token_id) {
    const Token& token = n_wt.token(token_id);
    assert(r_wt == nullptr || r_wt->token(token_id) == token);
    auto iter = retval.find(token.class_id);
    if (iter == retval.end()) {
      retval.insert(std::pair<ClassId, std::vector<float> >(token.class_id, std::vector<float>(n_wt.topic_size(), 0)));
      iter = retval.find(token.class_id);
    }

    for (int topic_id = 0; topic_id < n_wt.topic_size(); ++topic_id) {
      const float sum = n_wt.get(token_id, topic_id) + ((r_wt == nullptr) ? 0.0f : r_wt->get(token_id, topic_id));
      if (sum > 0)
        iter->second[topic_id] += sum;
    }
  }

  return retval;
}

static void FindPwtImpl(const PhiMatrix& n_wt, const PhiMatrix* r_wt, PhiMatrix* p_wt) {
  const int topic_size = n_wt.topic_size();
  const int token_size = n_wt.token_size();

  if (topic_size == 0 || token_size == 0) {
    LOG(WARNING) << "Attempt to calculate p_wt for empty matrix";
    return;
  }

  assert((r_wt == nullptr) || (r_wt->token_size() == n_wt.token_size() && r_wt->topic_size() == n_wt.topic_size()));
  assert(p_wt->token_size() == n_wt.token_size() && p_wt->topic_size() == n_wt.topic_size());

  std::map<ClassId, std::vector<float> > n_t = FindNormalizersImpl(n_wt, r_wt);
  for (int token_id = 0; token_id < token_size; ++token_id) {
    const Token& token = n_wt.token(token_id);
    assert(r_wt == nullptr || r_wt->token(token_id) == token);
    assert(p_wt->token(token_id) == token);
    const std::vector<float>& nt = n_t[token.class_id];
    for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
      if (nt[topic_index] <= 0)
        continue;

      float nwt_value = n_wt.get(token_id, topic_index);
      float rwt_value = (r_wt == nullptr) ? 0.0f : r_wt->get(token_id, topic_index);
      float value = std::max<float>(nwt_value + rwt_value, 0.0f) / nt[topic_index];
      if (value < 1e-16) {
        // Reset small values to 0.0 to avoid performance hit.
        // http://en.wikipedia.org/wiki/Denormal_number#Performance_issues
        // http://stackoverflow.com/questions/13964606/inconsistent-multiplication-performance-with-floats
        value = 0.0f;
      }

      p_wt->set(token_id, topic_index, value);
    }
  }
}

std::map<ClassId, std::vector<float> > PhiMatrixOperations::FindNormalizers(const PhiMatrix& n_wt) {
  return FindNormalizersImpl(n_wt, nullptr);
}

std::map<ClassId, std::vector<float> > PhiMatrixOperations::FindNormalizers(const PhiMatrix& n_wt,
                                                                            const PhiMatrix& r_wt) {
  return FindNormalizersImpl(n_wt, &r_wt);
}

void PhiMatrixOperations::FindPwt(const PhiMatrix& n_wt, PhiMatrix* p_wt) {
  FindPwtImpl(n_wt, nullptr, p_wt);
}

void PhiMatrixOperations::FindPwt(const PhiMatrix& n_wt, const PhiMatrix& r_wt, PhiMatrix* p_wt) {
  FindPwtImpl(n_wt, &r_wt, p_wt);
}

bool PhiMatrixOperations::HasEqualShape(const PhiMatrix& first, const PhiMatrix& second) {
  if (first.topic_size() != second.topic_size())
    return false;

  for (int i = 0; i < first.topic_size(); ++i)
    if (first.topic_name(i) != second.topic_name(i))
      return false;

  if (first.token_size() != second.token_size())
    return false;

  for (int i = 0; i < first.token_size(); ++i)
    if (first.token(i) != second.token(i))
      return false;

  return true;
}

}  // namespace core
}  // namespace artm
