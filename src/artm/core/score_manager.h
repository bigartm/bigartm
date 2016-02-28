// Copyright 2015, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_SCORE_MANAGER_H_
#define SRC_ARTM_CORE_SCORE_MANAGER_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

class ScoreManager : boost::noncopyable {
 public:
  ScoreManager() : lock_(), score_map_() {}

  void Append(std::shared_ptr<InstanceSchema> schema,
              const ScoreName& score_name, const std::string& score_blob);
  void Clear();
  bool RequestScore(std::shared_ptr<InstanceSchema> schema,
                    const ScoreName& score_name, ScoreData *score_data) const;
  void RequestAllScores(std::shared_ptr<InstanceSchema> schema,
                        ::google::protobuf::RepeatedPtrField< ::artm::ScoreData>* score_data) const;

 private:
  mutable boost::mutex lock_;
  std::map<ScoreName, std::shared_ptr<Score>> score_map_;
};

class ScoreTracker : boost::noncopyable {
 public:
  ScoreTracker() : lock_(), array_() {}
  void Clear();
  ScoreData* Add();
  void RequestScoreArray(const GetScoreArrayArgs& args, ScoreDataArray* score_data_array);

 private:
  mutable boost::mutex lock_;
  std::vector<std::shared_ptr<ScoreData>> array_;
};

}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_SCORE_MANAGER_H_
