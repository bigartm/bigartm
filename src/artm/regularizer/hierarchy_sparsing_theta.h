/* Copyright 2014, Additive Regularization of Topic Models.

Author: Nadia Chirkova (nadiinchi@gmail.com)
Based on code of Murat Apishev (great-mel@yandex.ru)
*/

#ifndef SRC_ARTM_REGULARIZER_HIERARCHY_SPARSING_THETA_H_
#define SRC_ARTM_REGULARIZER_HIERARCHY_SPARSING_THETA_H_
#include <memory>
#include <string>
#include <vector>

#include "artm/regularizer_interface.h"

namespace artm {
	namespace regularizer {

		class HierarchySparsingThetaAgent : public RegularizeThetaAgent {
		public:
			virtual void Apply(int item_index, int inner_iter, int topics_size, float* theta) const;

		private:
			friend class HierarchySparsingTheta;

			std::vector<float> topic_weight;  // tau for every topic
			std::vector<float> alpha_weight;  // iteration coef
			std::vector<float> topic_proportion;   // p(topic)
			std::vector<float> parent_topic_proportion;   // p(supertopic)
			double prior_parent_topic_probability;  // 1 / count(supertopics)
			bool regularization_on = true; // true if current batch is parent phi batch
		};

		class HierarchySparsingTheta : public RegularizerInterface {
		public:
			explicit HierarchySparsingTheta(const HierarchySparsingThetaConfig& config);

			virtual std::shared_ptr<RegularizeThetaAgent>
				CreateRegularizeThetaAgent(const Batch& batch, const ProcessBatchesArgs& args, double tau);

			virtual google::protobuf::RepeatedPtrField<std::string> topics_to_regularize();

			virtual bool Reconfigure(const RegularizerConfig& config);

		private:
			HierarchySparsingThetaConfig config_;
		};

	}  // namespace regularizer
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_HIERARCHY_SPARSING_THETA_H_
#pragma once
