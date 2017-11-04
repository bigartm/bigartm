// Copyright 2017, Additive Regularization of Topic Models.

// File 'common.h' contains constants, helpers and typedefs used across the entire library.
// The goal is to keep this file as short as possible.

#pragma once

#include <string>

#include "boost/lexical_cast.hpp"

#include "glog/logging.h"

#include "artm/core/exceptions.h"

#if defined(WIN32)
#pragma warning(push)
#pragma warning(disable: 4244 4267)
#include "artm/messages.pb.h"
#pragma warning(pop)
#else
#include "artm/messages.pb.h"
#endif

namespace artm {
namespace core {

typedef std::string ModelName;
typedef std::string ScoreName;
typedef std::string RegularizerName;
typedef std::string DictionaryName;
typedef std::string TopicName;

const int UnknownId = -1;

const std::string kBatchExtension = ".batch";

const int kIdleLoopFrequency = 1;  // 1 ms

const int kBatchNameLength = 6;

template <typename T>
std::string to_string(T value) {
  return boost::lexical_cast<std::string>(value);
}

}  // namespace core
}  // namespace artm
