// Copyright 2014, Additive Regularization of Topic Models.

#include "boost/thread.hpp"
#include "gtest/gtest.h"

#include "boost/filesystem.hpp"

#include "artm/cpp_interface.h"
#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/messages.pb.h"

#include "artm/core/internals.pb.h"

#include "artm_tests/test_mother.h"

#define ASSERT_APPROX_EQ(a, b) ASSERT_NEAR(a, b, (a + b) / 1e5)
void GenerateBatches(int batches_size, int nTokens, std::vector<std::shared_ptr< ::artm::Batch>>* batches);
std::string DescribeTopicModel(const ::artm::TopicModel& topic_model);
std::string DescribeThetaMatrix(const ::artm::ThetaMatrix& theta_matrix);
void CompareTopicModels(const ::artm::TopicModel& tm1, const ::artm::TopicModel& tm2, bool* ok);
void CompareThetaMatrices(const ::artm::ThetaMatrix& tm1, const ::artm::ThetaMatrix& tm2, bool *ok);

std::string runOfflineTest() {
  std::string target_path = artm::test::Helpers::getUniqueString();
  const int nTopics = 5;

  // Endpoints:
  // 5555 - master component (network_mode)
  // 5556 - node controller for workers (network_mode)

  ::artm::MasterComponentConfig master_config;
  master_config.set_cache_theta(true);
  master_config.set_processors_count(1);
  ::artm::MasterComponent master_component(master_config);

  // Create model
  artm::ModelConfig model_config;
  model_config.set_topics_count(nTopics);
  model_config.set_name(artm::test::Helpers::getUniqueString());
  artm::Model model(master_component, model_config);

  std::vector<std::shared_ptr< ::artm::Batch>> batches;
  int batches_size = 2;
  int nTokens = 10;
  GenerateBatches(batches_size, nTokens, &batches);
  for (int iter = 0; iter < 3; ++iter) {
    for (int iBatch = 0; iBatch < batches.size(); ++iBatch) {
      master_component.AddBatch(*batches[iBatch]);
    }

    master_component.WaitIdle();
    model.Synchronize(0.0);
  }

  std::shared_ptr< ::artm::TopicModel> topic_model = master_component.GetTopicModel(model.name());
  std::stringstream ss;
  ss << "Topic model:\n" << DescribeTopicModel(*topic_model);
  ss << "Theta matrix:\n";
  for (int i = 0; i < batches.size(); ++i) {
    ::artm::GetThetaMatrixArgs args;
    args.set_model_name(model.name());
    args.mutable_batch()->CopyFrom(*batches[i]);
    std::shared_ptr< ::artm::ThetaMatrix> theta_matrix = master_component.GetThetaMatrix(args);
    ss << DescribeThetaMatrix(*theta_matrix);
  }

  try { boost::filesystem::remove_all(target_path); }
  catch (...) {}

  return ss.str();
}

// artm_tests.exe --gtest_filter=RepeatableResult.Offline
TEST(RepeatableResult, Offline) {
  std::string first_result = runOfflineTest();
  std::string second_result = runOfflineTest();
  if (first_result != second_result)
    std::cout << first_result << "\n" << second_result;
  ASSERT_EQ(first_result, second_result);
}

// artm_tests.exe --gtest_filter=RepeatableResult.RandomGenerator
TEST(RepeatableResult, RandomGenerator) {
  int num = 10;
  size_t seed = 5;
  std::vector<float> first_result = ::artm::core::Helpers::GenerateRandomVector(num, seed);
  std::vector<float> second_result = ::artm::core::Helpers::GenerateRandomVector(num, seed);
  ASSERT_EQ(first_result.size(), num);
  ASSERT_EQ(second_result.size(), num);
  for (int i = 1; i < num; ++i) {
    ASSERT_EQ(first_result[i], second_result[i]);
    ASSERT_NE(first_result[i - 1], first_result[i]);
  }
}

void OverwriteTopicModel_internal(::artm::GetTopicModelArgs_RequestType request_type, bool use_sparse_format) {
  std::string target_path = artm::test::Helpers::getUniqueString();
  const int nTopics = 16;

  ::artm::MasterComponentConfig master_config;
  master_config.set_cache_theta(true);
  master_config.set_processors_count(1);
  ::artm::MasterComponent master_component(master_config);

  ::artm::RegularizerConfig sparse_phi_config;
  sparse_phi_config.set_name("sparse_phi");
  sparse_phi_config.set_type(::artm::RegularizerConfig_Type_SmoothSparsePhi);
  sparse_phi_config.set_config(::artm::SmoothSparsePhiConfig().SerializeAsString());
  ::artm::Regularizer sparse_phi(master_component, sparse_phi_config);

  // Create model
  artm::ModelConfig model_config;
  model_config.add_regularizer_name(sparse_phi_config.name());
  model_config.add_regularizer_tau(-0.05);
  for (int i = 0; i < nTopics; ++i) {
    std::stringstream ss;
    ss << "@topic_" << i;
    model_config.add_topic_name(ss.str());
  }

  model_config.set_name(artm::test::Helpers::getUniqueString());
  artm::Model model(master_component, model_config);

  std::vector<std::shared_ptr< ::artm::Batch>> batches;

  int batches_size = 2;
  int nTokens = 10;
  GenerateBatches(batches_size, nTokens, &batches);

  for (int iter = 0; iter < 3; ++iter) {
    for (int iBatch = 0; iBatch < batches.size(); ++iBatch) {
      master_component.AddBatch(*batches[iBatch]);
    }

    master_component.WaitIdle();
    model.Synchronize(0.0);
  }

  ::artm::MasterComponent master2(master_config);
  ::artm::Regularizer sparse_phi2(master2, sparse_phi_config);
  ::artm::Model model2(master2, model_config);

  ::artm::GetTopicModelArgs_RequestType request_types[2] = {
    artm::GetTopicModelArgs_RequestType_Pwt,
    artm::GetTopicModelArgs_RequestType_Nwt };

  bool pwt_request = request_type == artm::GetTopicModelArgs_RequestType_Pwt;
  bool nwt_request = !pwt_request;

  int slices = 3;
  for (int i = 0; i < slices; ++i) {
    ::artm::GetTopicModelArgs get_topic_model_args;
    get_topic_model_args.set_model_name(model.name());
    get_topic_model_args.set_request_type(request_type);
    get_topic_model_args.set_use_sparse_format(use_sparse_format);
    for (int topic_index = i; topic_index < nTopics; topic_index += slices) {
      get_topic_model_args.add_topic_name(model_config.topic_name(topic_index));
    }

    // To precisely overwrite topic model via n_wt counters one has manually
    // commit the model via model.Synchronize() method. This ensures that regularizers
    // are calculated for the model. Using "commit" argument gives wrong result
    // because it synchronizes model without regularizers. This is a reasonable
    // option when Overwrite is called through Pwt values, but for Nwt it is simply wrong.
    model2.Overwrite(*master_component.GetTopicModel(get_topic_model_args), false);
  }
  master2.WaitIdle();
  model2.Synchronize(/* decay_weight =*/ 0.0,
                      /* apply_weight =*/ 1.0,
                      /* invoke_regularizers =*/ nwt_request);  // invoke regularizers only for nwt_request

  bool ok = false;
  CompareTopicModels(*master2.GetTopicModel(model2.name()),
                      *master_component.GetTopicModel(model.name()), &ok);
  if (!ok) {
    std::cout << "New topic model:\n" << DescribeTopicModel(*master2.GetTopicModel(model2.name()));
    std::cout << "Old topic model:\n" << DescribeTopicModel(*master_component.GetTopicModel(model.name()));
  }
  ASSERT_TRUE(ok);
  for (int iBatch = 0; iBatch < batches.size(); ++iBatch) {
    CompareThetaMatrices(*master2.GetThetaMatrix(model.name(), *batches[iBatch]),
                          *master_component.GetThetaMatrix(model.name(), *batches[iBatch]), &ok);
  }
  ASSERT_TRUE(ok);

  if (pwt_request)
    return;  // do not validate further model inference for pwt_request

  // Run extra iteration and validate that model is stil still the same
  for (int iBatch = 0; iBatch < batches.size(); ++iBatch) {
    master_component.AddBatch(*batches[iBatch]);
    master2.AddBatch(*batches[iBatch]);
  }

  master_component.WaitIdle(); master2.WaitIdle();
  model.Synchronize(0.5); model2.Synchronize(0.5);

  CompareTopicModels(*master2.GetTopicModel(model2.name()),
                      *master_component.GetTopicModel(model.name()), &ok);
  ASSERT_TRUE(ok);
  for (int iBatch = 0; iBatch < batches.size(); ++iBatch) {
    CompareThetaMatrices(*master2.GetThetaMatrix(model.name(), *batches[iBatch]),
                          *master_component.GetThetaMatrix(model.name(), *batches[iBatch]), &ok);
    ASSERT_TRUE(ok);
  }

  try { boost::filesystem::remove_all(target_path); }
  catch (...) {}
}

// artm_tests.exe --gtest_filter=RepeatableResult.OverwriteTopicModel_Pwt_dense
TEST(RepeatableResult, OverwriteTopicModel_Pwt_dense) {
  OverwriteTopicModel_internal(artm::GetTopicModelArgs_RequestType_Pwt, false);
}

// artm_tests.exe --gtest_filter=RepeatableResult.OverwriteTopicModel_Pwt_sparse
TEST(RepeatableResult, OverwriteTopicModel_Pwt_sparse) {
  OverwriteTopicModel_internal(artm::GetTopicModelArgs_RequestType_Pwt, true);
}

// artm_tests.exe --gtest_filter=RepeatableResult.OverwriteTopicModel_Nwt_dense
TEST(RepeatableResult, OverwriteTopicModel_Nwt_dense) {
  OverwriteTopicModel_internal(artm::GetTopicModelArgs_RequestType_Nwt, false);
}

// artm_tests.exe --gtest_filter=RepeatableResult.OverwriteTopicModel_Nwt_sparse
TEST(RepeatableResult, OverwriteTopicModel_Nwt_sparse) {
  OverwriteTopicModel_internal(artm::GetTopicModelArgs_RequestType_Nwt, true);
}

void GenerateBatches(int batches_size, int nTokens, std::vector<std::shared_ptr< ::artm::Batch>>* batches) {
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
        field->add_token_count(1);
      }
    }

    batches->push_back(std::make_shared< ::artm::Batch>(batch));
  }
}

std::string DescribeTopicModel(const ::artm::TopicModel& topic_model) {
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

std::string DescribeThetaMatrix(const ::artm::ThetaMatrix& theta_matrix) {
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

void CompareTopicModels(const ::artm::TopicModel& tm1, const ::artm::TopicModel& tm2, bool* ok) {
  *ok = false;
  ASSERT_EQ(tm1.token_size(), tm2.token_size());
  ASSERT_EQ(tm1.token_weights_size(), tm2.token_weights_size());
  ASSERT_EQ(tm1.topic_index_size(), tm2.topic_index_size());
  if (tm1.topic_index_size() > 0)
    ASSERT_EQ(tm1.topic_index_size(), tm1.token_size());

  for (int i = 0; i < tm1.token_size(); ++i) {
    ASSERT_EQ(tm1.token(i), tm2.token(i));
    ASSERT_EQ(tm1.token_weights(i).value_size(), tm2.token_weights(i).value_size());
    for (int j = 0; j < tm1.token_weights(i).value_size(); ++j)
      ASSERT_APPROX_EQ(tm1.token_weights(i).value(j), tm2.token_weights(i).value(j));
    if (tm1.topic_index_size() > 0) {
      ASSERT_EQ(tm1.topic_index(i).value_size(), tm2.topic_index(i).value_size());
      for (int j = 0; j < tm1.topic_index(i).value_size(); ++j) {
        ASSERT_EQ(tm1.topic_index(i).value(j), tm2.topic_index(i).value(j));
      }
    }
  }
  *ok = true;
}

void CompareThetaMatrices(const ::artm::ThetaMatrix& tm1, const ::artm::ThetaMatrix& tm2, bool *ok) {
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
