// Copyright 2014, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#ifndef SRC_ARTM_CORE_TRANSFORM_FUNCTION_H_
#define SRC_ARTM_CORE_TRANSFORM_FUNCTION_H_

#include <memory>

#include "artm/core/common.h"

namespace artm {
namespace core {

// An interface for transformation function, used for KL-div regularizers.
class TransformFunction {
 public:
  static std::shared_ptr<TransformFunction> create(const TransformConfig& config);
  static std::shared_ptr<TransformFunction> create();
  virtual double apply(double value) = 0;
  virtual ~TransformFunction() { }
};

class LogarithmTransformFunction : public TransformFunction {
 public:
  virtual double apply(double value);
};

class PolynomialTransformFunction : public TransformFunction {
 public:
  PolynomialTransformFunction(double a, double n);
  virtual double apply(double value);

 private:
  double a_;
  double n_;
};

class ConstantTransformFunction : public TransformFunction {
 public:
  double apply(double value);
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_TRANSFORM_FUNCTION_H_
