// Copyright 2018, Additive Regularization of Topic Models.

#include "artm/core/processor_transaction_helpers.h"

namespace artm {
namespace core {

std::shared_ptr<BatchTransactionInfo> ProcessorTransactionHelpers::GetBatchTransactionsInfo(const Batch& batch) {
  ClassIdToTt class_id_to_tt;
  std::unordered_map<std::vector<int>, int, IntVectorHasher> transaction_ids_to_index;
  TransactionToIndex transaction_to_index;
  std::vector<std::vector<Token>> transactions;
  std::unordered_map<Token, int, TokenHasher> token_to_index;

  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    const Item& item = batch.item(item_index);

    for (int token_index = 0; token_index < item.transaction_start_index_size(); ++token_index) {
      const int start_index = item.transaction_start_index(token_index);
      const int end_index = (token_index + 1) < item.transaction_start_index_size() ?
                            item.transaction_start_index(token_index + 1) :
                            item.transaction_token_id_size();
      std::vector<int> vec;
      for (int i = start_index; i < end_index; ++i) {
        vec.push_back(item.transaction_token_id(i));
      }
      auto iter = transaction_ids_to_index.find(vec);
      if (iter == transaction_ids_to_index.end()) {
        transaction_ids_to_index.insert(std::make_pair(vec, transaction_ids_to_index.size()));

        std::string str;
        for (int token_id = start_index; token_id < end_index; ++token_id) {
          auto& tmp = batch.class_id(item.transaction_token_id(token_id));
          str += (token_id == start_index) ? tmp : TransactionSeparator + tmp;
        }

        TransactionType tt(str);
        for (const ClassId& class_id : tt.AsVector()) {
          class_id_to_tt[class_id].emplace(tt);
        }

        std::vector<Token> transaction;
        for (int idx = start_index; idx < end_index; ++idx) {
          const int token_id = item.transaction_token_id(idx);
          auto token = Token(batch.class_id(token_id), batch.token(token_id), tt);
          transaction.push_back(token);
          token_to_index.insert(std::make_pair(token, token_to_index.size()));
        }
        transactions.push_back(transaction);
        transaction_to_index.insert(std::make_pair(transaction, transaction_to_index.size()));
      }
    }
  }

  if (transaction_to_index.size() != transactions.size()) {
    LOG(ERROR) << "Fatal error: transaction_to_index.size() [ " << transaction_to_index.size()
      << " ] != transactions.size() [ " << transactions.size();
  }

  if (transaction_ids_to_index.size() != transactions.size()) {
    LOG(ERROR) << "Fatal error: transaction_ids_to_index.size() [ " << transaction_ids_to_index.size()
               << " ] != transactions.size() [ " << transactions.size();
  }

  return std::make_shared<BatchTransactionInfo>(
    BatchTransactionInfo(class_id_to_tt, transaction_ids_to_index,
                         transaction_to_index, transactions, token_to_index));
}

}  // namespace core
}  // namespace artm
