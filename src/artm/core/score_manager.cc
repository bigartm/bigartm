// Copyright 2017, Additive Regularization of Topic Models.

#include "artm/core/score_manager.h"

#include "boost/exception/diagnostic_information.hpp"

#include "glog/logging.h"

#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/instance.h"

namespace artm {
namespace core {

void ScoreManager::Append(const ScoreName& score_name,
                          const std::string& score_blob) {
  auto score_calculator = instance_->scores_calculators()->get(score_name);
  if (score_calculator == nullptr) {
    LOG(ERROR) << "Unable to find score calculator: " << score_name;
    return;
  }

  auto score_inc = score_calculator->CreateScore();
  if (!score_inc->ParseFromString(score_blob)) {
    LOG(ERROR) << "Merger was unable to parse score blob. The scores might be inacurate.";
    return;
  }

  // Note that the following operation must be atomic
  // (e.g. finding score / append score / setting score).
  // This is the reason to use explicit lock around score_map_ instead of ThreadSafeCollectionHolder.
  boost::lock_guard<boost::mutex> guard(lock_);
  auto iter = score_map_.find(score_name);
  if (iter != score_map_.end()) {
    score_calculator->AppendScore(*iter->second, score_inc.get());
    iter->second = score_inc;
  } else {
    score_map_.insert(std::pair<ScoreName, std::shared_ptr<Score>>(score_name, score_inc));
  }
}

void ScoreManager::Clear() {
  boost::lock_guard<boost::mutex> guard(lock_);
  score_map_.clear();
}

bool ScoreManager::RequestScore(const ScoreName& score_name,
                                ScoreData *score_data) const {
  auto score_calculator = instance_->scores_calculators()->get(score_name);
  if (score_calculator == nullptr) {
    BOOST_THROW_EXCEPTION(InvalidOperation(
      std::string("Attempt to request non-existing score: " + score_name)));
  }

  if (score_calculator->is_cumulative()) {
    boost::lock_guard<boost::mutex> guard(lock_);
    auto iter = score_map_.find(score_name);
    if (iter != score_map_.end()) {
      score_data->set_data(iter->second->SerializeAsString());
    } else {
      score_data->set_data(score_calculator->CreateScore()->SerializeAsString());
    }
  } else {
    std::shared_ptr<Score> score = score_calculator->CalculateScore();
    score_data->set_data(score->SerializeAsString());
  }

  score_data->set_type(score_calculator->score_type());
  score_data->set_name(score_name);
  return true;
}

void ScoreManager::RequestAllScores(::google::protobuf::RepeatedPtrField< ::artm::ScoreData>* score_data) const {
  if (score_data == nullptr) {
    return;
  }

  std::vector<ScoreName> score_names;
  {
    boost::lock_guard<boost::mutex> guard(lock_);
    for (const auto& elem : score_map_) {
      score_names.push_back(elem.first);
    }
  }

  for (auto& score_name : score_names) {
    ScoreData requested_score_data;
    if (RequestScore(score_name, &requested_score_data)) {
      score_data->Add()->Swap(&requested_score_data);
    }
  }
}

void ScoreManager::CopyFrom(const ScoreManager& score_manager) {
  boost::lock_guard<boost::mutex> guard(lock_);
  boost::lock_guard<boost::mutex> guard2(score_manager.lock_);
  score_map_ = score_manager.score_map_;
}

void ScoreTracker::Clear() {
  boost::lock_guard<boost::mutex> guard(lock_);
  array_.clear();
}

ScoreData* ScoreTracker::Add() {
  auto retval = std::make_shared<ScoreData>();

  boost::lock_guard<boost::mutex> guard(lock_);
  array_.push_back(retval);

  return retval.get();
}

void ScoreTracker::RequestScoreArray(const GetScoreArrayArgs& args, ScoreArray* score_array) {
  boost::lock_guard<boost::mutex> guard(lock_);
  for (auto& elem : array_) {
    if (elem->name() == args.score_name()) {
      score_array->add_score()->CopyFrom(*elem);
    }
  }
}

void ScoreTracker::CopyFrom(const ScoreTracker& score_tracker) {
  boost::lock_guard<boost::mutex> guard(lock_);
  boost::lock_guard<boost::mutex> guard2(score_tracker.lock_);
  array_ = score_tracker.array_;
}

}  // namespace core
}  // namespace artm
