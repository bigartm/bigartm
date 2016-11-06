#include <vector>
#include <algorithm>

#include "glog/logging.h"
#include "artm/core/protobuf_helpers.h"
#include "artm/utility/blas.h"

#include "artm/regularizer/topic_segmentation_ptdw.h"

namespace artm {
namespace regularizer {

void TopicSegmentationPtdwAgent::Apply(int item_index, int inner_iter, ::artm::utility::DenseMatrix<float>* ptdw) const {
    int local_token_size = ptdw->no_rows();
    int num_topics = ptdw->no_columns();
    std::vector<double> background_probability(local_token_size, 0.0);
    // if background topics are given, count probability for each word to be background
    if (config_.background_topic_names().size()) {
        std::vector<bool> is_background_topic = core::is_member(args_.topic_name(), config_.background_topic_names());
        for (int i = 0; i < local_token_size; ++i) {
            const float* local_ptdw_ptr = &(*ptdw)(i, 0);  // NOLINT
            for (int k = 0; k < num_topics; ++k) {
                if (is_background_topic[k]) {
                    background_probability[i] += local_ptdw_ptr[k];
                }
            }
        }
    }
    
    int h = config_.window();
    double threshold_topic_change = config_.threshold();
    ::artm::utility::DenseMatrix<float> copy_ptdw(*ptdw);
    std::vector<double> left_distribution(num_topics, 0.0);
    std::vector<double> right_distribution(num_topics, 0.0);
    double left_weights = 0.0;
    double right_weights = 0.0;
    int l_topic, r_topic;  //topic ids on which maximum of the distribution is reached
    bool changes_topic = false;
    int main_topic = std::distance(&(*ptdw)(0, 0), std::max_element(&(*ptdw)(0, 0), &(*ptdw)(0, num_topics)));  //NOLINT
    for (int k = 0; k < num_topics; ++k) {
        if (k == main_topic) {
            (*ptdw)(0, k) = 1;
        }
        else {
            (*ptdw)(0, k) = 0;
        }
    }
    for (int i = 0; i < h && i < local_token_size; ++i) {
        for (int k = 0; k < num_topics; ++k) {
            right_distribution[k] += copy_ptdw(i, k) * (1 - background_probability[i]);
        }
        right_weights += 1 - background_probability[i];
    }
    for (int i = 1; i < local_token_size; ++i) {
        for (int k = 0; k < num_topics; ++k) {
            left_distribution[k] += copy_ptdw(i - 1, k) * (1 - background_probability[i - 1]);
            right_distribution[k] -= copy_ptdw(i - 1, k) * (1 - background_probability[i - 1]);
        }
        left_weights += 1 - background_probability[i - 1];
        right_weights -= 1 - background_probability[i - 1];
        if (i <= local_token_size - h) {
            for (int k = 0; k < num_topics; ++k) {
                right_distribution[k] += copy_ptdw(i + h - 1, k) * (1 - background_probability[i + h - 1]);
            }
            right_weights += 1 - background_probability[i + h - 1];
        }
        if (i > h) {
            for (int k = 0; k < num_topics; ++k) {
                left_distribution[k] -= copy_ptdw(i - h - 1, k) * (1 - background_probability[i - h - 1]);
            }
            left_weights -= 1 - background_probability[i - h - 1];
        }
        l_topic = std::distance(left_distribution.begin(), std::max_element(left_distribution.begin(), left_distribution.end()));
        r_topic = std::distance(right_distribution.begin(), std::max_element(right_distribution.begin(), right_distribution.end()));
        changes_topic = ((left_distribution[l_topic]/left_weights - right_distribution[l_topic]/right_weights) / 2 + (right_distribution[r_topic]/right_weights - left_distribution[r_topic]/left_weights) / 2 > threshold_topic_change);
        if (changes_topic) {
            main_topic = r_topic;
        }
        for (int k = 0; k < num_topics; ++k) {
            if (k == main_topic) {
                (*ptdw)(i, k) = 1;
            }
            else {
                (*ptdw)(i, k) = 0;
            }
        }
    }
}
    
std::shared_ptr<RegularizePtdwAgent>
TopicSegmentationPtdw::CreateRegularizePtdwAgent(const Batch& batch, const ProcessBatchesArgs& args, double tau) {
    TopicSegmentationPtdwAgent* agent = new TopicSegmentationPtdwAgent(config_, args, tau);
    std::shared_ptr<RegularizePtdwAgent> retval(agent);
    return retval;
}
    
bool TopicSegmentationPtdw::Reconfigure(const RegularizerConfig& config) {
    std::string config_blob = config.config();
    TopicSegmentationPtdwConfig regularizer_config;
    if (!regularizer_config.ParseFromString(config_blob)) {
        BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException("Unable to parse TopicSegmentationPtdwConfig from RegularizerConfig.config"));
    }
    config_.CopyFrom(regularizer_config);
    return true;
}

} //namespace regularizer
} //namespace artm