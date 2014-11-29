// Copyright 2014, Additive Regularization of Topic Models.

#include "artm_tests/test_mother.h"

namespace artm {
namespace test {

ModelConfig TestMother::GenerateModelConfig() const {
  ModelConfig config;
  config.set_enabled(true);
  config.set_topics_count(nTopics);
  config.add_regularizer_name(regularizer_name);
  ::artm::core::ModelName model_name =
    boost::lexical_cast<std::string>(boost::uuids::random_generator()());
  config.set_name(boost::lexical_cast<std::string>(model_name));
  return config;
}

RegularizerConfig TestMother::GenerateRegularizerConfig() const {
  ::artm::SmoothSparseThetaConfig regularizer_1_config;
  for (int i = 0; i < 12; ++i)
    regularizer_1_config.add_alpha_iter(0.8);

  ::artm::RegularizerConfig general_regularizer_1_config;
  general_regularizer_1_config.set_name(regularizer_name);
  general_regularizer_1_config.set_type(artm::RegularizerConfig_Type_SmoothSparseTheta);
  general_regularizer_1_config.set_config(regularizer_1_config.SerializeAsString());

  return general_regularizer_1_config;
}

}  // namespace test
}  // namespace artm
