// Copyright 2015, Additive Regularization of Topic Models.

#include "artm/core/phi_matrix_operations.h"

#include <assert.h>

#include <algorithm>
#include <utility>

#include "artm/core/protobuf_helpers.h"

namespace artm {
namespace core {

void PhiMatrixOperations::RetrieveExternalTopicModel(const PhiMatrix& phi_matrix,
                                                     const ::artm::GetTopicModelArgs& get_model_args,
                                                     ::artm::TopicModel* topic_model) {
  const bool use_sparse_format = get_model_args.use_sparse_format();

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

    if (!use_sparse_format) {
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

}  // namespace core
}  // namespace artm
