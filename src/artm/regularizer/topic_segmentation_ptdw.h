#ifndef SRC_ARTM_REGULARIZER_TOPIC_SEGMENTATION_PTDW_H_
#define SRC_ARTM_REGULARIZER_TOPIC_SEGMENTATION_PTDW_H_

#include <string>
#include <vector>

#include "artm/regularizer_interface.h"

namespace artm {
namespace regularizer {

class TopicSegmentationPtdwAgent : public RegularizePtdwAgent {
private:
    friend class TopicSegmentationPtdw;
    TopicSegmentationPtdwConfig config_;
    ProcessBatchesArgs args_;
    double tau_;

public:
    TopicSegmentationPtdwAgent(const TopicSegmentationPtdwConfig& config, const ProcessBatchesArgs& args, double tau)
      : config_(config), args_(args), tau_(tau) {}

    virtual void Apply(int item_index, int inner_iter, ::artm::utility::DenseMatrix<float>* ptdw) const;
};

class TopicSegmentationPtdw : public RegularizerInterface {
public:
    explicit TopicSegmentationPtdw(const TopicSegmentationPtdwConfig& config)
    : config_(config) {}

    virtual std::shared_ptr<RegularizePtdwAgent>
    CreateRegularizePtdwAgent(const Batch& batch, const ProcessBatchesArgs& args, double tau);

    virtual bool Reconfigure(const RegularizerConfig& config);

private:
    TopicSegmentationPtdwConfig config_;
};

}  // namespace regularizer
}  // namespace artm

#endif  // SRC_ARTM_REGULARIZER_TOPIC_SEGMENTATION_PTDW_H_
