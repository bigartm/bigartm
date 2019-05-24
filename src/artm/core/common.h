// Copyright 2018, Additive Regularization of Topic Models.

// File 'common.h' contains constants, helpers and typedefs used across the entire library.
// The goal is to keep this file as short as possible.

#pragma once

#include <string>

#include "boost/lexical_cast.hpp"

#include "glog/logging.h"

#include "artm/core/exceptions.h"

#include "artm/artm_export.h"

#if defined(_MSC_VER)
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

// Defined in 3rdparty/protobuf-3.0.0/src/google/protobuf/io/coded_stream.h
const int64_t kProtobufCodedStreamTotalBytesLimit = 2147483647ULL;

const std::string TokenCoocFrequency = "tf";
const std::string DocumentCoocFrequency = "df";

const std::string kParentPhiMatrixBatch = "__parent_phi_matrix_batch__";

template <typename T>
std::string to_string(T value) {
  return boost::lexical_cast<std::string>(value);
}

}  // namespace core
}  // namespace artm
