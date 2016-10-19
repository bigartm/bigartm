// Copyright 2015, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_DICTIONARY_OPERATIONS_H_
#define SRC_ARTM_CORE_DICTIONARY_OPERATIONS_H_

#include <memory>
#include <string>

#include "artm/core/dictionary.h"

namespace artm {
namespace core {

// DictionaryOperations contains helper methods to operate on Dictionary class.
class DictionaryOperations {
 public:
  static std::shared_ptr<Dictionary> Create(const DictionaryData& data);

  static void Export(const ExportDictionaryArgs& args, ThreadSafeDictionaryCollection* dictionaries);

  static std::shared_ptr<Dictionary> Import(const ImportDictionaryArgs& args);

  static std::shared_ptr<Dictionary> Gather(const GatherDictionaryArgs& args,
    const ThreadSafeCollectionHolder<std::string, Batch>& mem_batches);

  static std::shared_ptr<Dictionary> Filter(const FilterDictionaryArgs& args,
    ThreadSafeDictionaryCollection* dictionaries);

  static void StoreIntoDictionaryData(std::shared_ptr<Dictionary> dictionary, DictionaryData* data);
};

}  // namespace core
}  // namespace artm

#endif  // SRC_ARTM_CORE_DICTIONARY_OPERATIONS_H_
