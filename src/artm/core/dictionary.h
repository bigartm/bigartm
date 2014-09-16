// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_DICTIONARY_H_
#define SRC_ARTM_CORE_DICTIONARY_H_

#include <string>
#include <map>

#include "artm/messages.pb.h"
#include "artm/core/common.h"
#include "artm/core/thread_safe_holder.h"

namespace artm {
namespace core {

typedef std::map<Token, ::artm::DictionaryEntry> DictionaryMap;
typedef ThreadSafeCollectionHolder<std::string, DictionaryMap> ThreadSafeDictionaryCollection;

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_DICTIONARY_H_
