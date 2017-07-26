// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <chrono>  // NOLINT
#include <string>

#include "boost/functional/hash.hpp"

#include "glog/logging.h"

namespace artm {
namespace core {

// CuckooWatch class is a utility that measures time from creation until destruction.
// It is named after a Cuckoo bird (https://en.wikipedia.org/wiki/Cuckoo).
class CuckooWatch {
 public:
  explicit CuckooWatch(std::string message)
      : message_(message), submessage_(), start_(std::chrono::system_clock::now()), parent_(nullptr),
        threshold_ms_(0) { }
  explicit CuckooWatch(std::string message, int threshold_ms)
      : message_(message), submessage_(), start_(std::chrono::system_clock::now()), parent_(nullptr),
        threshold_ms_(threshold_ms) { }
  CuckooWatch(std::string message, CuckooWatch* parent)
      : message_(message), submessage_(), start_(std::chrono::system_clock::now()), parent_(parent),
        threshold_ms_(1) { }
  CuckooWatch(std::string message, CuckooWatch* parent, int threshold_ms)
      : message_(message), submessage_(), start_(std::chrono::system_clock::now()), parent_(parent),
        threshold_ms_(threshold_ms) { }

  ~CuckooWatch() {
    auto delta = (std::chrono::system_clock::now() - start_);
    auto delta_ms = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
    if (delta_ms.count() < threshold_ms_) {
      return;
    }

    if (parent_ == nullptr) {
      std::stringstream ss;
      ss << delta_ms.count() << "ms in " << message_;
      if (!submessage_.empty()) {
        ss << " [including " << submessage_ << "]";
      }
      LOG(INFO) << ss.str();
    } else {
      std::stringstream ss;
      ss << delta_ms.count() << "ms in " << message_ << "; ";
      parent_->submessage_ += ss.str();
    }
  }

 private:
  std::string message_;
  std::string submessage_;
  std::chrono::time_point<std::chrono::system_clock> start_;
  CuckooWatch* parent_;
  int threshold_ms_;
};

}  // namespace core
}  // namespace artm
