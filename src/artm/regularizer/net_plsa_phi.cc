// Copyright 2017, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include <string>
#include <vector>

#include "artm/core/protobuf_helpers.h"
#include "artm/core/phi_matrix.h"
#include "artm/regularizer/net_plsa_phi.h"

namespace artm {
namespace regularizer {

bool NetPlsaPhi::RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                                    const ::artm::core::PhiMatrix& n_wt,
                                    ::artm::core::PhiMatrix* result) {
  // read the parameters from config and control their correctness
  const int topic_size = p_wt.topic_size();
  const int token_size = p_wt.token_size();

  std::vector<bool> topics_to_regularize;
  if (config_.topic_name().size() == 0)
    topics_to_regularize.assign(topic_size, true);
  else
    topics_to_regularize = core::is_member(p_wt.topic_name(), config_.topic_name());


  if (!config_.has_class_id()) {
    LOG(ERROR) << "There's no name of vertex modality in class_id field for" <<
                  "NetPLSA regularizer. Cancel it's launch.";
    return false;
  }

  bool has_weights = config_.vertex_weight_size();
  if (has_weights && config_.vertex_name_size() != config_.vertex_weight_size()) {
    LOG(ERROR) << "Non-empty vertex_weight array should have the same length " <<
                  "with vertex_name array in NetPLSA regularizer config";
  }

  std::vector<double> n_t(topic_size, 0.0);
  for (int token_id = 0; token_id < token_size; ++token_id) {
    const auto& token = n_wt.token(token_id);
    if (token.class_id != config_.class_id())
      continue;

    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      if (topics_to_regularize[topic_id]) {
        n_t[topic_id] += static_cast<double>(n_wt.get(token_id, topic_id));
      }
    }
  }

  for (int token_id = 0; token_id < token_size; ++token_id) {
    const auto& token_u = p_wt.token(token_id);
    if (token_u.class_id != config_.class_id())
      continue;

    auto name_iter = vertex_name2id_.find(token_u.keyword);
    if (name_iter == vertex_name2id_.end())
      continue;

    auto edge_iter = edge_weights_.find(name_iter->second);
    if (edge_iter == edge_weights_.end())
      continue;

    float D_u = has_weights ? config_.vertex_weight(name_iter->second) : 1.0;
    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      if (!topics_to_regularize[topic_id])
        continue;

      float value = 0.0;
      float p_ut = p_wt.get(token_id, topic_id);
      for (auto iter = edge_iter->second.begin(); iter != edge_iter->second.end(); ++iter) {
        auto it = id2vertex_name_.find(iter->first);
        if (it == id2vertex_name_.end()) {
          LOG(WARNING) << "Edge links to vertex " << iter->first <<
                          ", that does not exist in list of vertices, it will be skipped";
          continue;
        }

        float p_vt = p_wt.get(p_wt.token_index(
            ::artm::core::Token(config_.class_id(), it->second)), topic_id);
        float D_v = has_weights ? config_.vertex_weight(iter->first) : 1.0;

        value += iter->second * (p_vt / (D_u * D_v) - p_ut / (D_u * D_u));
      }
      value *= n_t[topic_id] * n_t[topic_id];
      result->set(token_id, topic_id, value);
    }
  }

  return true;
}

google::protobuf::RepeatedPtrField<std::string> NetPlsaPhi::topics_to_regularize() {
  return config_.topic_name();
}

google::protobuf::RepeatedPtrField<std::string> NetPlsaPhi::class_ids_to_regularize() {
  google::protobuf::RepeatedPtrField<std::string> retval;
  std::string* ptr = retval.Add();
  *ptr = config_.class_id();
  return retval;
}

void NetPlsaPhi::UpdateNetInfo(const NetPlsaPhiConfig& config) {
  config_.clear_first_vertex_index();
  config_.clear_second_vertex_index();
  config_.clear_edge_weight();
  config_.clear_vertex_name();

  vertex_name2id_.clear();
  id2vertex_name_.clear();
  int num_vertices = config.vertex_name_size();
  if (num_vertices) {
    for (int i = 0; i < num_vertices; ++i) {
      vertex_name2id_[config.vertex_name(i)] = i;
      id2vertex_name_[i] = config.vertex_name(i);
    }
  }

  edge_weights_.clear();
  int num_edges = config.first_vertex_index_size();
  if (num_edges) {
    if (num_edges != config.second_vertex_index_size() || num_edges != config.edge_weight_size()) {
      BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
          "Both vertex indices and value arrays should have the same length"));
    }

    const auto func = [&](int first_index, int second_index, float value) {  // NOLINT
      auto iter = edge_weights_.find(first_index);
      if (iter == edge_weights_.end()) {
        edge_weights_.insert(std::make_pair(first_index, std::unordered_map<int, float>()));
        iter = edge_weights_.find(first_index);
      }
      iter->second.insert(std::make_pair(second_index, value));
    };

    for (int i = 0; i < num_edges; ++i) {
      func(config.first_vertex_index(i), config.second_vertex_index(i), config.edge_weight(i));
      if (!config.symmetric_edge_weights())
        continue;

      func(config.second_vertex_index(i), config.first_vertex_index(i), config.edge_weight(i));
    }
  }
}

bool NetPlsaPhi::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  NetPlsaPhiConfig regularizer_config;
  if (!regularizer_config.ParseFromString(config_blob)) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse SmoothSparsePhiConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  UpdateNetInfo(regularizer_config);

  return true;
}

}  // namespace regularizer
}  // namespace artm
