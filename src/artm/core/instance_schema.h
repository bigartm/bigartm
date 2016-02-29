// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_INSTANCE_SCHEMA_H_
#define SRC_ARTM_CORE_INSTANCE_SCHEMA_H_

#include <map>
#include <memory>
#include <vector>
#include <string>

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"

namespace artm {

class RegularizerInterface;
class ScoreCalculatorInterface;

namespace core {

class InstanceSchema {
 public:
  InstanceSchema();
  explicit InstanceSchema(const InstanceSchema& schema);
  std::shared_ptr<InstanceSchema> Duplicate() const;
  void RequestMasterComponentInfo(MasterComponentInfo* master_info) const;

  std::shared_ptr<RegularizerInterface> regularizer(const std::string& name) const;
  void set_regularizer(const std::string& name,
                       const std::shared_ptr<RegularizerInterface>& regularizer);
  bool has_regularizer(const std::string& name) const;
  void clear_regularizer(const std::string& name);
  std::shared_ptr<std::vector<std::string> > regularizers_list();

  std::shared_ptr<ScoreCalculatorInterface> score_calculator(const ScoreName& name) const;
  void set_score_calculator(const ScoreName& name,
                            const std::shared_ptr<ScoreCalculatorInterface>& score_calculator);
  bool has_score_calculator(const ScoreName& name) const;
  void clear_score_calculator(const ScoreName& name);
  void clear_score_calculators();

 private:
  std::map<std::string, std::shared_ptr<RegularizerInterface> > regularizers_;
  std::map<ScoreName, std::shared_ptr<ScoreCalculatorInterface>> score_calculators_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_INSTANCE_SCHEMA_H_
