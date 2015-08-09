// Copyright 2014, Additive Regularization of Topic Models.

#include "artm_tests/test_mother.h"

#include <vector>

#include "gtest/gtest.h"

#include "artm/cpp_interface.h"

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

void TestMother::GenerateBatches(int batches_size, int nTokens,
                                 std::vector<std::shared_ptr< ::artm::Batch>>* batches) {
  srand(1);
  for (int iBatch = 0; iBatch < batches_size; ++iBatch) {
    ::artm::Batch batch;
    batch.set_id(artm::test::Helpers::getUniqueString());

    // Same dictionary across all batches
    for (int i = 0; i < nTokens; i++) {
      std::stringstream str;
      str << "token" << i;
      batch.add_token(str.str());
    }

    artm::Item* item = batch.add_item();
    item->set_id(iBatch);  // one item per batch
    artm::Field* field = item->add_field();
    for (int iToken = 0; iToken < nTokens; ++iToken) {
      if (iToken == 0 || rand() % 3 == 0) {  // NOLINT
        field->add_token_id(iToken);
        field->add_token_weight(1.0);
      }
    }

    batches->push_back(std::make_shared< ::artm::Batch>(batch));
  }
}

std::string Helpers::DescribeTopicModel(const ::artm::TopicModel& topic_model) {
  std::stringstream ss;
  for (int i = 0; i < topic_model.token_size(); ++i) {
    ss << topic_model.token(i) << ": ";
    for (int j = 0; j < topic_model.topic_name_size(); ++j) {
      ss << topic_model.token_weights(i).value(j) << " ";
    }
    ss << std::endl;
  }

  return ss.str();
}

std::string Helpers::DescribeThetaMatrix(const ::artm::ThetaMatrix& theta_matrix) {
  std::stringstream ss;
  for (int i = 0; i < theta_matrix.item_id_size(); ++i) {
    ss << theta_matrix.item_id(i) << ": ";
    for (int j = 0; j < theta_matrix.topics_count(); ++j) {
      ss << theta_matrix.item_weights(i).value(j) << " ";
    }
    ss << std::endl;
  }

  return ss.str();
}

void Helpers::CompareTopicModels(const ::artm::TopicModel& tm1, const ::artm::TopicModel& tm2, bool* ok) {
  *ok = false;
  ASSERT_EQ(tm1.token_size(), tm2.token_size());
  ASSERT_EQ(tm1.token_weights_size(), tm2.token_weights_size());
  ASSERT_EQ(tm1.topic_index_size(), tm2.topic_index_size());
  if (tm1.topic_index_size() > 0)
    ASSERT_EQ(tm1.topic_index_size(), tm1.token_size());

  for (int i = 0; i < tm1.token_size(); ++i) {
    ASSERT_EQ(tm1.token(i), tm2.token(i));
    ASSERT_EQ(tm1.token_weights(i).value_size(), tm2.token_weights(i).value_size());
    for (int j = 0; j < tm1.token_weights(i).value_size(); ++j) {
      float tm1_value = tm1.token_weights(i).value(j);
      float tm2_value = tm2.token_weights(i).value(j);
      if (fabs(tm1_value) < 1e-12) tm1_value = 0.0f;
      if (fabs(tm2_value) < 1e-12) tm2_value = 0.0f;
      ASSERT_APPROX_EQ(tm1_value, tm2_value);
    }
    if (tm1.topic_index_size() > 0) {
      ASSERT_EQ(tm1.topic_index(i).value_size(), tm2.topic_index(i).value_size());
      for (int j = 0; j < tm1.topic_index(i).value_size(); ++j) {
        ASSERT_EQ(tm1.topic_index(i).value(j), tm2.topic_index(i).value(j));
      }
    }
  }
  *ok = true;
}

void Helpers::CompareThetaMatrices(const ::artm::ThetaMatrix& tm1, const ::artm::ThetaMatrix& tm2, bool *ok) {
  *ok = false;
  ASSERT_EQ(tm1.item_id_size(), tm2.item_id_size());
  ASSERT_EQ(tm1.item_weights_size(), tm2.item_weights_size());
  ASSERT_EQ(tm1.topic_index_size(), tm2.topic_index_size());
  if (tm1.topic_index_size() > 0)
    ASSERT_EQ(tm1.topic_index_size(), tm1.item_id_size());

  for (int i = 0; i < tm1.item_id_size(); ++i) {
    ASSERT_EQ(tm1.item_id(i), tm2.item_id(i));
    ASSERT_EQ(tm1.item_weights(i).value_size(), tm2.item_weights(i).value_size());
    for (int j = 0; j < tm1.item_weights(i).value_size(); ++j)
      ASSERT_APPROX_EQ(tm1.item_weights(i).value(j), tm2.item_weights(i).value(j));
    if (tm1.topic_index_size() > 0) {
      ASSERT_EQ(tm1.topic_index(i).value_size(), tm2.topic_index(i).value_size());
      for (int j = 0; j < tm1.topic_index(i).value_size(); ++j) {
        ASSERT_EQ(tm1.topic_index(i).value(j), tm2.topic_index(i).value(j));
      }
    }
  }
  *ok = true;
}

void TestMother::GenerateBatches(int batches_size, int nTokens, const std::string& target_folder) {
  std::vector<std::shared_ptr< ::artm::Batch>> batches;
  GenerateBatches(batches_size, nTokens, &batches);
  for (int i = 0; i < batches.size(); ++i)
    artm::SaveBatch(*batches[i], target_folder);
}

}  // namespace test
}  // namespace artm
