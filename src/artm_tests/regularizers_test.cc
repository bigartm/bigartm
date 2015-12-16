// Copyright 2014, Additive Regularization of Topic Models.

#include <memory>

#include "gtest/gtest.h"

#include "artm/messages.pb.h"

#include "artm/cpp_interface.h"
#include "artm/core/instance.h"

#include "artm_tests/test_mother.h"

// artm_tests.exe --gtest_filter=Regularizers.TopicSelection
TEST(Regularizers, TopicSelection) {
  int nTopics = 10;

  // create master
  auto master_config = artm::MasterComponentConfig();
  master_config.set_cache_theta(true);
  artm::MasterComponent master(master_config);

  // create regularizer
  ::artm::RegularizerConfig regularizer_config;
  regularizer_config.set_name("TopicSelectionRegularizer");
  regularizer_config.set_type(::artm::RegularizerConfig_Type_TopicSelectionTheta);

  ::artm::TopicSelectionThetaConfig internal_config;
  for (int i = 0; i < nTopics; ++i)
    internal_config.add_topic_value(static_cast<float>(i) / nTopics);

  regularizer_config.set_config(internal_config.SerializeAsString());
  ::artm::Regularizer regularizer_topic_selection(master, regularizer_config);

  // create model
  artm::ModelConfig model_config;
  model_config.set_topics_count(nTopics);
  model_config.add_regularizer_name("TopicSelectionRegularizer");
  model_config.add_regularizer_tau(0.5f);
  model_config.set_name("model_config");
  artm::Model model(master, model_config);

  // iterations
  std::vector<std::shared_ptr< ::artm::Batch> > batches;
  ::artm::test::TestMother::GenerateBatches(1, 5, &batches);
  for (int iter = 0; iter < 3; ++iter) {
    for (unsigned iBatch = 0; iBatch < batches.size(); ++iBatch) {
      master.AddBatch(*batches[iBatch]);
    }
    master.WaitIdle();
    model.Synchronize(0.0);
  }

  // get and check theta
  artm::GetThetaMatrixArgs args;
  args.set_model_name(model.name().c_str());
  std::shared_ptr< ::artm::ThetaMatrix> theta_matrix = master.GetThetaMatrix(args);

  ASSERT_NEAR(theta_matrix->item_weights(0).value(0), 0.43774, 0.00001);
  ASSERT_NEAR(theta_matrix->item_weights(0).value(1), 0.24127, 0.00001);
  ASSERT_NEAR(theta_matrix->item_weights(0).value(2), 0.15803, 0.00001);
  ASSERT_NEAR(theta_matrix->item_weights(0).value(3), 0.08591, 0.00001);
  ASSERT_NEAR(theta_matrix->item_weights(0).value(4), 0.03449, 0.00001);
  ASSERT_NEAR(theta_matrix->item_weights(0).value(5), 0.02333, 0.00001);
  ASSERT_NEAR(theta_matrix->item_weights(0).value(6), 0.01110, 0.00001);
  ASSERT_NEAR(theta_matrix->item_weights(0).value(7), 0.00458, 0.00001);
  ASSERT_NEAR(theta_matrix->item_weights(0).value(8), 0.00245, 0.00001);
  ASSERT_NEAR(theta_matrix->item_weights(0).value(9), 0.00107, 0.00001);
}
