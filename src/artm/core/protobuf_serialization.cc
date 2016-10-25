// Copyright 2016, Additive Regularization of Topic Models.

#include "artm/core/protobuf_serialization.h"

#include "google/protobuf/util/json_util.h"

#include "glog/logging.h"

#include "artm/core/exceptions.h"

namespace pb = ::google::protobuf;

namespace artm {
namespace core {

void ProtobufSerialization::ParseFromString(const std::string& string, google::protobuf::Message* message) {
  if (use_json_format_) {
    VLOG(0) << string;
    if (pb::util::JsonStringToMessage(string, message) != pb::util::Status::OK)
      BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException("Unable to parse the message from json format"));
    return;
  } else {
    if (!message->ParseFromString(string))
      BOOST_THROW_EXCEPTION(::artm::core::CorruptedMessageException("Unable to parse the message"));
  }
}

void ProtobufSerialization::ParseFromArray(const char* buffer, int length, google::protobuf::Message* message) {
  ParseFromString((length >= 0) ? std::string(buffer, length) : std::string(buffer), message);
}

std::string ProtobufSerialization::SerializeAsString(const google::protobuf::Message& message) {
  std::string retval;
  SerializeToString(message, &retval);
  return retval;
}

void ProtobufSerialization::SerializeToString(const google::protobuf::Message& message, std::string* output) {
  if (use_json_format_) {
    output->clear();
    if (pb::util::MessageToJsonString(message, output) != pb::util::Status::OK)
      BOOST_THROW_EXCEPTION(::artm::core::InvalidOperation("Unable to serialize the message to json format"));
    VLOG(0) << *output;
  } else {
    if (!message.SerializeToString(output))
      BOOST_THROW_EXCEPTION(::artm::core::InvalidOperation("Unable to serialize the message"));
  }
}

}  // namespace core
}  // namespace artm
