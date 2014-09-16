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
class ModelConfig;
class ScoreCalculatorInterface;

namespace core {

class InstanceSchema {
 public:
  InstanceSchema();
  explicit InstanceSchema(const InstanceSchema& schema);
  explicit InstanceSchema(const MasterComponentConfig& config);

  const MasterComponentConfig& config() const;
  void set_config(const MasterComponentConfig& config);

  const ModelConfig& model_config(ModelName id) const;
  void set_model_config(ModelName id, const std::shared_ptr<const ModelConfig>& model_config);
  bool has_model_config(ModelName id) const;
  void clear_model_config(ModelName id);

  std::shared_ptr<RegularizerInterface> regularizer(const std::string& name);
  void set_regularizer(const std::string& name,
                       const std::shared_ptr<RegularizerInterface>& regularizer);
  bool has_regularizer(const std::string& name) const;
  void clear_regularizer(const std::string& name);

  std::shared_ptr<ScoreCalculatorInterface> score_calculator(const ScoreName& name);
  void set_score_calculator(const ScoreName& name,
                            const std::shared_ptr<ScoreCalculatorInterface>& score_calculator);
  bool has_score_calculator(const ScoreName& name) const;
  void clear_score_calculator(const ScoreName& name);
  void clear_score_calculators();

  std::vector<ModelName> GetModelNames() const;

 private:
  MasterComponentConfig config_;
  std::map<std::string, std::shared_ptr<RegularizerInterface> > regularizers_;
  std::map<ModelName, std::shared_ptr<const ModelConfig> > models_config_;
  std::map<ScoreName, std::shared_ptr<ScoreCalculatorInterface>> score_calculators_;
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_INSTANCE_SCHEMA_H_
