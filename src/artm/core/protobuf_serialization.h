// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <stdint.h>

#include <string>

#include "google/protobuf/message.h"

namespace artm {
namespace core {

// A signleton class that manages all serialization to and from protobuf messages
class ProtobufSerialization {
 public:
  static ProtobufSerialization& singleton() {
    static ProtobufSerialization instance;
    return instance;
  }

  void SetFormatToJson() { use_json_format_ = true; }
  void SetFormatToBinary() { use_json_format_ = false; }
  bool IsJson() { return use_json_format_; }
  bool IsBinary() { return !use_json_format_; }

  void ParseFromString(const std::string& string, google::protobuf::Message* message);
  void ParseFromArray(const char* buffer, int64_t length, google::protobuf::Message* message);
  void SerializeToString(const google::protobuf::Message& message, std::string* output);
  std::string SerializeAsString(const google::protobuf::Message& message);

  template<typename T>
  static std::string ConvertJsonToBinary(const std::string& json) {
    T temporary;
    return ConvertJsonToBinary(json, &temporary);
  }

  template<typename T>
  static std::string ConvertBinaryToJson(const std::string& binary) {
    T temporary;
    return ConvertBinaryToJson(binary, &temporary);
  }

  static std::string ConvertJsonToBinary(const std::string& json, google::protobuf::Message* temporary);
  static std::string ConvertBinaryToJson(const std::string& binary, google::protobuf::Message* temporary);

 private:
  ProtobufSerialization() : use_json_format_(false) { }
  bool use_json_format_;
};

}  // namespace core
}  // namespace artm
