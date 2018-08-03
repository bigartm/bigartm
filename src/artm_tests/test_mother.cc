// Copyright 2017, Additive Regularization of Topic Models.

#include "artm_tests/test_mother.h"

#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"

#include "artm/cpp_interface.h"
#include "artm/core/helpers.h"

namespace artm {
namespace test {

namespace fs = boost::filesystem;

artm::Batch Helpers::GenerateBatch(int nTokens, int nDocs,
                                   const std::string& class1, const std::string& class2) {
  artm::Batch batch;
  batch.set_id("11972762-6a23-4524-b089-7122816aff72");
  for (int i = 0; i < nTokens; i++) {
    std::stringstream str;
    str << "token" << i;
    std::string class_id = (i % 2 == 0) ? class1 : class2;
    batch.add_token(str.str());
    batch.add_class_id(class_id);
  }

  for (int iDoc = 0; iDoc < nDocs; iDoc++) {
    artm::Item* item = batch.add_item();
    item->set_id(iDoc);
    for (int iToken = 0; iToken < nTokens; ++iToken) {
      item->add_token_id(iToken);
      item->add_transaction_start_index(item->transaction_start_index_size());
      int background_count = (iToken > 40) ? (1 + rand() % 5) : 0;  // NOLINT
      int topical_count = ((iToken < 40) && ((iToken % 10) == (iDoc % 10))) ? 10 : 0;
      item->add_token_weight(static_cast<float>(background_count + topical_count));
    }
    item->add_transaction_start_index(item->transaction_start_index_size());
  }

  return batch;
}

artm::DictionaryData Helpers::GenerateDictionary(int nTokens, const std::string& class1, const std::string& class2) {
  ::artm::DictionaryData dictionary_data;
  for (int i = 0; i < nTokens; i++) {
    std::stringstream str;
    str << "token" << i;
    std::string class_id = (i % 2 == 0) ? class1 : class2;
    if (class_id.empty()) {
      continue;
    }
    dictionary_data.add_token(str.str());
    dictionary_data.add_class_id(class_id);
  }
  return dictionary_data;
}

void Helpers::ConfigurePerplexityScore(const std::string& score_name,
                                       artm::MasterModelConfig* master_config,
                                       const std::vector<std::string>& class_ids,
                                       const std::vector<std::string>& tt_names) {
  ::artm::ScoreConfig score_config;
  ::artm::PerplexityScoreConfig perplexity_config;
  for (const auto& c : class_ids) {
    perplexity_config.add_class_id(c);
  }

  for (const auto& s : tt_names) {
    perplexity_config.add_transaction_typename(s);
  }
  score_config.set_config(perplexity_config.SerializeAsString());
  score_config.set_type(::artm::ScoreType_Perplexity);
  score_config.set_name(score_name);
  master_config->add_score_config()->CopyFrom(score_config);
}

MasterModelConfig TestMother::GenerateMasterModelConfig(int nTopics) {
  MasterModelConfig config;
  for (int i = 0; i < nTopics; ++i) {
    config.add_topic_name("Topic" + boost::lexical_cast<std::string>(i));
  }
  ::artm::core::ModelName model_name =
    boost::lexical_cast<std::string>(boost::uuids::random_generator()());
  config.set_pwt_name(boost::lexical_cast<std::string>(model_name));
  return config;
}

RegularizerConfig TestMother::GenerateRegularizerConfig() const {
  ::artm::SmoothSparseThetaConfig regularizer_1_config;
  for (int i = 0; i < 12; ++i) {
    regularizer_1_config.add_alpha_iter(0.8f);
  }

  ::artm::RegularizerConfig general_regularizer_1_config;
  general_regularizer_1_config.set_name(regularizer_name);
  general_regularizer_1_config.set_type(artm::RegularizerType_SmoothSparseTheta);
  general_regularizer_1_config.set_config(regularizer_1_config.SerializeAsString());

  return general_regularizer_1_config;
}

std::vector<std::shared_ptr< ::artm::Batch>>
TestMother::GenerateBatches(int batches_size, int nTokens, ::artm::DictionaryData* dictionary) {
  std::vector<std::shared_ptr< ::artm::Batch>> retval;
  bool first_iter = true;
  for (int iBatch = 0; iBatch < batches_size; ++iBatch) {
    ::artm::Batch batch;
    batch.set_id(artm::test::Helpers::getUniqueString());

    // Same dictionary across all batches
    for (int i = 0; i < nTokens; i++) {
      std::stringstream str;
      str << "token" << i;
      batch.add_token(str.str());

      if (dictionary != nullptr && first_iter) {
        dictionary->add_token(str.str());
      }
    }

    artm::Item* item = batch.add_item();
    item->set_id(iBatch);  // one item per batch
    for (int iToken = 0; iToken < nTokens; ++iToken) {
      const int somewhat_random = iToken + iBatch + (iToken + 1)*(iBatch + 1);
      if (iToken == 0 || somewhat_random % 3 == 0) {  // NOLINT
        item->add_token_id(iToken);
        item->add_transaction_start_index(item->transaction_start_index_size());
        item->add_token_weight(1.0);
      }
    }
    item->add_transaction_start_index(item->transaction_start_index_size());

    retval.push_back(std::make_shared< ::artm::Batch>(batch));
    first_iter = false;
  }
  return retval;
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
    for (int j = 0; j < theta_matrix.num_topics(); ++j) {
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
  ASSERT_EQ(tm1.topic_indices_size(), tm2.topic_indices_size());
  if (tm1.topic_indices_size() > 0) {
    ASSERT_EQ(tm1.topic_indices_size(), tm1.token_size());
  }

  for (int i = 0; i < tm1.token_size(); ++i) {
    ASSERT_EQ(tm1.token(i), tm2.token(i));
    ASSERT_EQ(tm1.token_weights(i).value_size(), tm2.token_weights(i).value_size());
    for (int j = 0; j < tm1.token_weights(i).value_size(); ++j) {
      float tm1_value = tm1.token_weights(i).value(j);
      float tm2_value = tm2.token_weights(i).value(j);
      if (fabs(tm1_value) < 1e-12) {
        tm1_value = 0.0f;
      }
      if (fabs(tm2_value) < 1e-12) {
        tm2_value = 0.0f;
      }
      ASSERT_APPROX_EQ(tm1_value, tm2_value);
    }
    if (tm1.topic_indices_size() > 0) {
      ASSERT_EQ(tm1.topic_indices(i).value_size(), tm2.topic_indices(i).value_size());
      for (int j = 0; j < tm1.topic_indices(i).value_size(); ++j) {
        ASSERT_EQ(tm1.topic_indices(i).value(j), tm2.topic_indices(i).value(j));
      }
    }
  }
  *ok = true;
}

void Helpers::CompareThetaMatrices(const ::artm::ThetaMatrix& tm1, const ::artm::ThetaMatrix& tm2, bool *ok) {
  *ok = false;
  ASSERT_EQ(tm1.item_id_size(), tm2.item_id_size());
  ASSERT_EQ(tm1.item_weights_size(), tm2.item_weights_size());
  ASSERT_EQ(tm1.topic_indices_size(), tm2.topic_indices_size());
  if (tm1.topic_indices_size() > 0) {
    ASSERT_EQ(tm1.topic_indices_size(), tm1.item_id_size());
  }

  for (int i = 0; i < tm1.item_id_size(); ++i) {
    ASSERT_EQ(tm1.item_id(i), tm2.item_id(i));
    ASSERT_EQ(tm1.item_weights(i).value_size(), tm2.item_weights(i).value_size());
    for (int j = 0; j < tm1.item_weights(i).value_size(); ++j) {
      ASSERT_APPROX_EQ(tm1.item_weights(i).value(j), tm2.item_weights(i).value(j));
    }
    if (tm1.topic_indices_size() > 0) {
      ASSERT_EQ(tm1.topic_indices(i).value_size(), tm2.topic_indices(i).value_size());
      for (int j = 0; j < tm1.topic_indices(i).value_size(); ++j) {
        ASSERT_EQ(tm1.topic_indices(i).value(j), tm2.topic_indices(i).value(j));
      }
    }
  }
  *ok = true;
}

fs::path Helpers::getTestDataDir() {
  auto dir = std::getenv("BIGARTM_UNITTEST_DATA");
  // Construct path object only once.
  // Since C++11 local static variable initialization is thread-safe.
  static const fs::path testDataDir = dir != nullptr ? fs::path(dir) : fs::path("../../../test_data");
  return testDataDir;
}

void TestMother::GenerateBatches(int batches_size, int nTokens, const std::string& target_folder) {
  auto batches = GenerateBatches(batches_size, nTokens);
  for (unsigned i = 0; i < batches.size(); ++i) {
    artm::core::Helpers::SaveBatch(*batches[i], target_folder, batches[i]->id());
  }
}

}  // namespace test
}  // namespace artm
