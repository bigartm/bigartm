// Copyright 2014, Additive Regularization of Topic Models.

#include "artm/core/scores_merger.h"

#include "boost/exception/diagnostic_information.hpp"

#include "glog/logging.h"

#include "artm/core/exceptions.h"
#include "artm/core/helpers.h"
#include "artm/core/instance_schema.h"

namespace artm {
namespace core {

void ScoresMerger::Append(std::shared_ptr<InstanceSchema> schema,
                          const ModelName& model_name, const ScoreName& score_name,
                          const std::string& score_blob) {
  auto key = std::make_pair(model_name, score_name);
  auto score_calculator = schema->score_calculator(score_name);
  if (score_calculator == nullptr) {
    LOG(ERROR) << "Unable to find score calculator: " << score_name;
    return;
  }

  auto score_inc = score_calculator->CreateScore();
  if (!score_inc->ParseFromString(score_blob)) {
    LOG(ERROR) << "Merger was unable to parse score blob. The scores might be inacurate.";
    return;
  }

  boost::lock_guard<boost::mutex> guard(lock_);
  auto iter = score_map_.find(key);
  if (iter != score_map_.end()) {
    score_calculator->AppendScore(*iter->second, score_inc.get());
    iter->second = score_inc;
  } else {
    score_map_.insert(std::pair<ScoreKey, std::shared_ptr<Score>>(key, score_inc));
  }
}

void ScoresMerger::ResetScores(const ModelName& model_name) {
  boost::lock_guard<boost::mutex> guard(lock_);
  if (model_name.empty()) {
    score_map_.clear();
    return;
  }

  // Scott Meyers, Effective STL. Item 9 - Choose carefully among erasing options.
  for (auto iter = score_map_.begin(); iter != score_map_.end(); /* nothing */) {
    if (iter->first.first == model_name) {
      score_map_.erase(iter++);
    } else {
      ++iter;
    }
  }
}

bool ScoresMerger::RequestScore(std::shared_ptr<InstanceSchema> schema,
                                const ModelName& model_name, const ScoreName& score_name,
                                ScoreData *score_data) const {
  auto score_calculator = schema->score_calculator(score_name);
  if (score_calculator == nullptr)
    BOOST_THROW_EXCEPTION(InvalidOperation("Attempt to request non-existing score"));

  if (!score_calculator->is_cumulative())
    return false;

  {
    boost::lock_guard<boost::mutex> guard(lock_);
    auto iter = score_map_.find(ScoreKey(model_name, score_name));
    if (iter != score_map_.end()) {
      score_data->set_data(iter->second->SerializeAsString());
    } else {
      score_data->set_data(score_calculator->CreateScore()->SerializeAsString());
    }
  }

  score_data->set_type(score_calculator->score_type());
  score_data->set_name(score_name);
  return true;
}

}  // namespace core
}  // namespace artm
