// Copyright 2015, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_SCORES_MERGER_H_
#define SRC_ARTM_CORE_SCORES_MERGER_H_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "boost/thread/locks.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"

#include "artm/messages.pb.h"
#include "artm/core/common.h"
#include "artm/core/instance_schema.h"
#include "artm/core/thread_safe_holder.h"
#include "artm/score_calculator_interface.h"

namespace artm {
namespace core {

class ScoresMerger : boost::noncopyable {
 public:
  explicit ScoresMerger() : lock_(), score_map_() {}

  void Append(std::shared_ptr<InstanceSchema> schema,
              const ModelName& model_name, const ScoreName& score_name, const std::string& score_blob);
  void ResetScores(const ModelName& model_name);
  bool RequestScore(std::shared_ptr<InstanceSchema> schema,
                    const ModelName& model_name, const ScoreName& score_name, ScoreData *score_data) const;

 private:
  mutable boost::mutex lock_;

  // Map from model name and score name to the score
  typedef std::pair<ModelName, ScoreName> ScoreKey;
  std::map<ScoreKey, std::shared_ptr<Score>> score_map_;
};

}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_SCORES_MERGER_H_
