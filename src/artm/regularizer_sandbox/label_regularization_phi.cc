// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#include "artm/regularizer_sandbox/label_regularization_phi.h"

#include <string>
#include <vector>

#include "artm/core/regularizable.h"
#include "artm/core/topic_model.h"

namespace artm {
namespace regularizer_sandbox {

bool LabelRegularizationPhi::RegularizePhi(::artm::core::Regularizable* topic_model, double tau) {
  // read the parameters from config and control their correctness
  const int topic_size = topic_model->topic_size();
  const int token_size = topic_model->token_size();

  auto topic_name = topic_model->topic_name();
  std::vector<bool> topics_to_regularize;
  if (config_.topic_name_size() > 0) {
    for (int i = 0; i < topic_size; ++i)
      topics_to_regularize.push_back(false);

    for (int topic_id = 0; topic_id < config_.topic_name_size(); ++topic_id) {
      for (int real_topic_id = 0; real_topic_id < topic_size; ++real_topic_id) {
        if (topic_name.Get(real_topic_id) == config_.topic_name(topic_id)) {
          topics_to_regularize[real_topic_id] = true;
          break;
        }
      }
    }
  } else {
    for (int i = 0; i < topic_size; ++i)
      topics_to_regularize.push_back(true);
  }

  bool use_all_classes = false;
  std::vector<artm::core::ClassId> classes_to_regularize;
  if (config_.class_id_size() > 0) {
    for (auto& class_id : config_.class_id())
      classes_to_regularize.push_back(class_id);
  } else {
    use_all_classes = true;
  }

  bool has_dictionary = true;
  if (!config_.has_dictionary_name()) {
    has_dictionary = false;
  }

  auto dictionary_ptr = dictionary(config_.dictionary_name());
  if (has_dictionary && dictionary_ptr == nullptr) {
    has_dictionary = false;
  }

  core::TokenCollectionWeights p_wt(topic_model->topic_size());
  topic_model->FindPwt(&p_wt);
  std::map<core::ClassId, std::vector<float> >& n_t = topic_model->FindNormalizers();

  std::vector<float> p_c_coeffs;
  if (!has_dictionary) {
    for (int token_id = 0; token_id < token_size; ++token_id) {
      if (!use_all_classes) {
        if (std::find(classes_to_regularize.begin(),
                      classes_to_regularize.end(),
                      topic_model->token(token_id).class_id)
            != classes_to_regularize.end()) {
          p_c_coeffs.push_back(1);
        }
      } else {
        p_c_coeffs.push_back(1);
      }
    }
  } else {
    for (auto& entry_iter = dictionary_ptr->begin(); entry_iter != dictionary_ptr->end(); ++entry_iter)
      p_c_coeffs.push_back(entry_iter->second.value());
  }
    
  // proceed the regularization
  int useful_token_id = 0;
  for (int token_id = 0; token_id < token_size; ++token_id) {
    bool regularize_this_token = false;
    auto token_class_id = topic_model->token(token_id).class_id;
    if (!use_all_classes) {
      if (std::find(classes_to_regularize.begin(),
                    classes_to_regularize.end(),
                    token_class_id)
          != classes_to_regularize.end()) {
        regularize_this_token = true;
      }
    } else {
      regularize_this_token = true;
    }
    if (regularize_this_token) {
      // count sum of weights
      float weights_sum = 0.0f;
      for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
        if (topics_to_regularize[topic_id]) {
          // token_class_id is anyway presented in n_t
          weights_sum += p_wt[token_id][topic_id] * n_t.find(token_class_id)->second[topic_id];
        }
      }
      // form the value
      for (int topic_id = 0; topic_id < topic_size; ++topic_id) {
        if (topics_to_regularize[topic_id]) {
          float p_c = p_c_coeffs[useful_token_id++];
          float weight = p_wt[token_id][topic_id] * n_t.find(token_class_id)->second[topic_id];
          float value = static_cast<float>(p_c * tau * weight / weights_sum);
          topic_model->IncreaseRegularizerWeight(token_id, topic_id, value);
        }
      }
    }
  }

  return true;
}

bool LabelRegularizationPhi::Reconfigure(const RegularizerConfig& config) {
  std::string config_blob = config.config();
  SmoothSparsePhiConfig regularizer_config;
  if (!regularizer_config.ParseFromArray(config_blob.c_str(), config_blob.length())) {
    BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException(
      "Unable to parse SmoothSparsePhiConfig from RegularizerConfig.config"));
  }

  config_.CopyFrom(regularizer_config);
  return true;
}

}  // namespace regularizer_sandbox
}  // namespace artm
