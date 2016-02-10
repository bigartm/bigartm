// Copyright 2015, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_SERIALIZER_H_
#define SRC_ARTM_CORE_SERIALIZER_H_

#include <string>

#include "boost/serialization/serialization.hpp"
#include "boost/serialization/version.hpp"
#include "boost/serialization/split_free.hpp"

#include "artm/core/common.h"
#include "artm/core/dictionary.h"

namespace boost {
namespace serialization {

template<class Archive>
void save(Archive& ar, const ::artm::core::Token& obj, const unsigned int version) {  // NOLINT
  ar << obj.keyword;
  ar << obj.class_id;
}

template<class Archive>
void load(Archive& ar, ::artm::core::Token& obj, const unsigned int version) {  // NOLINT
  std::string keyword;
  ::artm::core::ClassId class_id;

  ar >> keyword;
  ar >> class_id;
  obj = ::artm::core::Token(class_id, keyword);
}

template<class Archive>
void serialize(Archive& ar, ::artm::core::DictionaryEntry& obj, const unsigned int version) {  // NOLINT
  ar & obj.token_;
  ar & obj.token_value_;
  ar & obj.token_tf_;
  ar & obj.token_df_;
}

}  // namespace serialization
}  // namespace boost

// namespace artm {
// namespace core {

BOOST_CLASS_VERSION( ::artm::core::Token, 0)  // NOLINT
BOOST_SERIALIZATION_SPLIT_FREE(::artm::core::Token)  // NOLINT
BOOST_CLASS_VERSION( ::artm::core::DictionaryEntry, 0)  // NOLINT

// }  // namesppace core
// }  // namespace artm

#endif  // SRC_ARTM_CORE_SERIALIZER_H_
