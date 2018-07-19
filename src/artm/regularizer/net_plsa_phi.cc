// Copyright 2017, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include "artm/core/protobuf_helpers.h"
#include "artm/core/phi_matrix.h"
#include "artm/core/phi_matrix_operations.h"
#include "artm/core/token.h"

#include "artm/regularizer/net_plsa_phi.h"

namespace artm {
namespace regularizer {

bool NetPlsaPhi::RegularizePhi(const ::artm::core::PhiMatrix& p_wt,
                               const ::artm::core::PhiMatrix& n_wt,
                               ::artm::core::PhiMatrix* result) {
  if (!::artm::core::PhiMatrixOperations::HasEqualShape(p_wt, n_wt)) {
    LOG(ERROR) << "NetPlsaPhi does not support changes in p_wt and n_wt matrix. Cancel it's launch.";
    return false;
  }

  // read the parameters from config and control their correctness
  const int topic_size = p_wt.topic_size();

  std::vector<bool> topics_to_regularize;
  if (config_.topic_name().size() == 0) {
    topics_to_regularize.assign(topic_size, true);
  } else {
    topics_to_regularize = core::is_member(p_wt.topic_name(), config_.topic_name());
  }

  if (!config_.has_class_id()) {
    LOG(ERROR) << "There's no name of vertex modality in class_id field for" <<
                  "NetPLSA regularizer. Cancel it's launch.";
    return false;
  }
  const auto& class_id = config_.class_id();

  bool has_weights = config_.vertex_weight_size();
  if (has_weights && vertex_name_.size() != config_.vertex_weight_size()) {
    LOG(ERROR) << "Non-empty vertex_weight array should have the same length " <<
                  "with vertex_name array in NetPLSA regularizer config (" <<
                  vertex_name_.size() << " != " << config_.vertex_weight_size() << ")";
  }

  auto normalizers = artm::core::PhiMatrixOperations::FindNormalizers(n_wt);
  auto norm_iter = normalizers.find(class_id);
  if (norm_iter == normalizers.end()) {
    LOG(ERROR) << "NetPlsaPhiConfig.class_id " << class_id
               << " does not exists in n_wt matrix. Cancel regularization.";
  }
  const auto& n_t = norm_iter->second;

  for (int vertex_id = 0; vertex_id < vertex_name_.size(); ++vertex_id) {
    auto edge_iter = edge_weights_.find(vertex_id);
    if (edge_iter == edge_weights_.end()) {
      continue;
    }

    const int token_id = p_wt.token_index(::artm::core::Token(class_id, vertex_name_[vertex_id]));
    if (token_id < 0) {
      continue;
    }

    float D_u = has_weights ? config_.vertex_weight(vertex_id) : 1.0;
    for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
      if (!topics_to_regularize[topic_id]) {
        continue;
      }

      float value = 0.0;
      float p_ut = p_wt.get(token_id, topic_id);

      for (const auto& pair_id : edge_iter->second) {
        if (pair_id.first >= vertex_name_.size() || pair_id.first < 0) {
          LOG(WARNING) << "Edge links to vertex " << pair_id.first <<
                          ", that does not exist in list of vertices, it will be skipped";
          continue;
        }

        const int index = p_wt.token_index(::artm::core::Token(class_id, vertex_name_[pair_id.first]));
        if (index < 0) {
          continue;
        }
        float p_vt = p_wt.get(index, topic_id);
        float D_v = has_weights ? config_.vertex_weight(pair_id.first) : 1.0;

        value += pair_id.second * (p_vt / D_v - p_ut / D_u) * (1 / D_u);
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

  vertex_name_.clear();
  int num_vertices = config.vertex_name_size();
  for (int i = 0; i < num_vertices; ++i) {
    vertex_name_.push_back(config.vertex_name(i));
  }

  edge_weights_.clear();
  int num_edges = config.first_vertex_index_size();
  if (num_edges) {
    if (num_edges != config.second_vertex_index_size() || num_edges != config.edge_weight_size()) {
      std::stringstream ss;
      ss << "Both vertex indices and value arrays should have the same length " << num_edges << ", now: "
         << config.second_vertex_index_size() << " and " << config.edge_weight_size();
      BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(ss.str()));
    }

    for (int i = 0; i < num_edges; ++i) {
      edge_weights_[config.first_vertex_index(i)][config.second_vertex_index(i)] = config.edge_weight(i);
      if (!config.symmetric_edge_weights()) {
        continue;
      }

      edge_weights_[config.second_vertex_index(i)][config.first_vertex_index(i)] = config.edge_weight(i);
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
