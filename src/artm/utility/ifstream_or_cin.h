// Copyright 2015, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_UTILITY_IFSTREAM_OR_CIN_H_
#define SRC_ARTM_UTILITY_IFSTREAM_OR_CIN_H_

#include <string>
#include <iostream>  // NOLINT

#include "boost/filesystem.hpp"
#include "boost/iostreams/device/mapped_file.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/throw_exception.hpp"

using boost::iostreams::mapped_file_source;

namespace artm {
namespace utility {

class ifstream_or_cin {
 public:
  explicit ifstream_or_cin(const std::string& filename) {
    if (filename == "-")  // read from std::cin
      return;

    if (!boost::filesystem::exists(filename))
      BOOST_THROW_EXCEPTION(::artm::core::DiskReadException("File " + filename + " does not exist."));

    if (boost::filesystem::exists(filename) && !boost::filesystem::is_regular_file(filename))
      BOOST_THROW_EXCEPTION(::artm::core::DiskReadException(
                  "File " + filename + " is not regular (probably it's a directory)."));

    file_.open(filename);
  }

  std::istream& get_stream() { return file_.is_open() ? file_ : std::cin; }

 private:
  boost::iostreams::stream<mapped_file_source> file_;
};

}  // namespace utility
}  // namespace artm

#endif  // SRC_ARTM_UTILITY_IFSTREAM_OR_CIN_H_
