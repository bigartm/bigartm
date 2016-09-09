// Copyright 2016, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_UTILITY_PROGRESS_PRINTER_H_
#define SRC_ARTM_UTILITY_PROGRESS_PRINTER_H_

#include <cstddef>

namespace artm {
namespace utility {

class ProgressPrinter {
 public:
  explicit ProgressPrinter(size_t max) noexcept;
  void Add(int delta) noexcept;
  void Set(size_t pos) noexcept;
  size_t max() const noexcept { return max_; }

 private:
  size_t max_;
  size_t pos_;
};

}  // namespace utility
}  // namespace artm

#endif  // SRC_ARTM_UTILITY_PROGRESS_PRINTER_H_
