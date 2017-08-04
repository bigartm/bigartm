// Copyright 2017, Additive Regularization of Topic Models.

// Author: Murat Apishev (great-mel@yandex.ru)

#pragma once

#include <memory>

#include "artm/core/common.h"

namespace artm {
namespace core {

// An interface for transformation function, used for KL-div regularizers.
class TransformFunction {
 public:
  static std::shared_ptr<TransformFunction> create(const TransformConfig& config);
  static std::shared_ptr<TransformFunction> create();
  virtual float apply(float value) = 0;
  virtual ~TransformFunction() { }
};

class LogarithmTransformFunction : public TransformFunction {
 public:
  virtual float apply(float value);
};

class PolynomialTransformFunction : public TransformFunction {
 public:
  PolynomialTransformFunction(float a, float n);
  virtual float apply(float value);

 private:
  float a_;
  float n_;
};

class ConstantTransformFunction : public TransformFunction {
 public:
  float apply(float value);
};

}  // namespace core
}  // namespace artm
