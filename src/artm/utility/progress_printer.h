// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <cstddef>

namespace artm {
namespace utility {

class ProgressPrinter {
 public:
  explicit ProgressPrinter(size_t max);
  void Add(int delta);
  void Set(size_t pos);
  size_t max() const { return max_; }

 private:
  size_t max_;
  size_t pos_;
};

}  // namespace utility
}  // namespace artm
