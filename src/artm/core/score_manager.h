// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "boost/thread/locks.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"

#include "artm/core/common.h"
#include "artm/core/thread_safe_holder.h"
#include "artm/score_calculator_interface.h"

namespace artm {
namespace core {

class Instance;

// ScoreManager class stores and aggregates theta scores.
// Its implementation is thread safe because it can be called simultaneoudly from multiple processor threads.
class ScoreManager : boost::noncopyable {
 public:
  explicit ScoreManager(Instance* instance) : instance_(instance), lock_(), score_map_() { }

  void Append(const ScoreName& score_name, const std::string& score_blob);
  void Clear();
  bool RequestScore(const ScoreName& score_name, ScoreData *score_data) const;
  void RequestAllScores(::google::protobuf::RepeatedPtrField< ::artm::ScoreData>* score_data) const;
  void CopyFrom(const ScoreManager& score_manager);

 private:
  Instance* instance_;
  mutable boost::mutex lock_;
  std::map<ScoreName, std::shared_ptr<Score>> score_map_;
};

// ScoreTracker class stores historical data for each score
// (which is particularly important for Online algorithm to see the history of all scores within one iteration).
// It is used to implement RequestScoreArray API.
// The purpose of this class is solely to store the scores --- not to merge them.
// This class stores both Phi-scores (non-cumulative) and Theta-scores (cumulative).
class ScoreTracker : boost::noncopyable {
 public:
  ScoreTracker() : lock_(), array_() { }
  void Clear();
  ScoreData* Add();
  void RequestScoreArray(const GetScoreArrayArgs& args, ScoreArray* score_data_array);
  void CopyFrom(const ScoreTracker& score_tracker);
  const std::vector<std::shared_ptr<ScoreData>>& GetDataUnsafe() const { return array_; }
  size_t Size() const { return array_.size(); }

 private:
  mutable boost::mutex lock_;
  std::vector<std::shared_ptr<ScoreData>> array_;
};

}  // namespace core
}  // namespace artm
