// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_CORE_PROCESSOR_H_
#define SRC_ARTM_CORE_PROCESSOR_H_

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/bind.hpp"
#include "boost/utility.hpp"

#include "artm/messages.pb.h"

#include "artm/core/common.h"
#include "artm/core/internals.pb.h"
#include "artm/core/thread_safe_holder.h"

namespace artm {
namespace core {

class InstanceSchema;
class Merger;
class TopicModel;
class TopicWeightIterator;

template<typename T>
class Matrix;

class Processor : boost::noncopyable {
 public:
  Processor(ThreadSafeQueue<std::shared_ptr<const ProcessorInput> >*  processor_queue,
            ThreadSafeQueue<std::shared_ptr<const ModelIncrement> >* merger_queue,
            const Merger& merger,
            const ThreadSafeHolder<InstanceSchema>& schema);

  ~Processor();

  void FindThetaMatrix(const Batch& batch, const ModelName& model_name, ThetaMatrix* theta_matrix);

 private:
  ThreadSafeQueue<std::shared_ptr<const ProcessorInput> >* processor_queue_;
  ThreadSafeQueue<std::shared_ptr<const ModelIncrement> >* merger_queue_;
  const Merger& merger_;
  const ThreadSafeHolder<InstanceSchema>& schema_;

  mutable std::atomic<bool> is_stopping;
  boost::thread thread_;

  void ThreadFunction();

  // Helper class to iterate on stream
  class StreamIterator : boost::noncopyable {
   public:
    // Iterates on a global stream (all items in the batch)
    explicit StreamIterator(const ProcessorInput& processor_input);

    const Item* Next();
    const Item* Current() const;

    // Checks whether Current() item belongs to a specific stream
    bool InStream(const std::string& stream_name);
    bool InStream(int stream_index);

    inline int item_index() const { return item_index_; }

   private:
    int items_count_;
    int item_index_;
    const Mask* stream_flags_;
    const ProcessorInput& processor_input_;
  };

  // Helper class to perform the actual job
  // (inferring theta distribution or perplexity calculation)
  class ItemProcessor : boost::noncopyable {
   public:
    explicit ItemProcessor(const TopicModel& topic_model,
                           const std::vector<Token>& token_dict,
                           const std::map<ClassId, float>& class_id_to_weight,
                           std::shared_ptr<InstanceSchema> schema);

    void InferTheta(const ModelConfig& model,
                    const Item& item,
                    ModelIncrement* model_increment,
                    bool update_model,
                    float* theta);

   private:
    const TopicModel& topic_model_;
    const std::vector<Token>& token_dict_;
    const std::map<ClassId, float>& class_id_to_weight_;
    std::shared_ptr<InstanceSchema> schema_;
  };

  // Helper class to iterate through tokens in one item
  class TokenIterator : boost::noncopyable {
   public:
    enum Mode {
      Mode_Known = 1,
      Mode_Unknown = 2,
      Mode_KnownAndUnknown = 3
    };

    TokenIterator(const std::vector<Token>& token_dict,
                  const std::map<ClassId, float>& class_id_to_weight,
                  const TopicModel& topic_model,
                  const Item& item,
                  const std::string& field_name,
                  Mode mode = Mode_KnownAndUnknown);

    bool Next();
    void Reset();

    const Token& token() const { return token_; }
    float token_class_weight() const { return token_class_weight_; }
    int id_in_model() const { return id_in_model_; }
    int id_in_batch() const { return id_in_batch_; }
    int count() const { return count_; }
    TopicWeightIterator GetTopicWeightIterator() const;

   private:
    const std::vector<Token>& token_dict_;
    const std::map<ClassId, float>& class_id_to_weight_;
    const TopicModel& topic_model_;
    const Field* field_;
    int token_size_;
    bool iterate_known_;
    bool iterate_unknown_;
    bool use_model_class_list_;

    // Current state of the iterator
    int token_index_;
    Token token_;
    float token_class_weight_;
    int id_in_model_;
    int id_in_batch_;
    int count_;
  };
};

template<typename T>
class Matrix {
 public:
  Matrix(int no_rows = 0, int no_columns = 0, bool store_by_rows = true)
    : no_rows_(no_rows),
      no_columns_(no_columns),
      store_by_rows_(store_by_rows),
      data_(nullptr) {
    if (no_rows > 0 && no_columns > 0) {
      data_ = new T[no_rows_ * no_columns_];
    }
  }

  Matrix(const Matrix<T>& src_matrix) {
    no_rows_ = src_matrix.no_rows();
    no_columns_ = src_matrix.no_columns();
    store_by_rows_ = src_matrix.store_by_rows_;
    data_ = new T[no_rows_ * no_columns_];
    for (int i = 0; i < no_rows_ * no_columns_; ++i) {
      data_[i] = src_matrix.get_data()[i];
    }
  }

  ~Matrix() {
    delete[] data_;
  }

  void InitializeZeros() {
    memset(data_, 0, sizeof(T) * no_rows_ * no_columns_);
  }

  T& operator() (int index_row, int index_col) {
    if (store_by_rows_) {
      return data_[index_row * no_columns_ + index_col];
    }
    return data_[index_col * no_rows_ + index_row];
  }

  const T& operator() (int index_row, int index_col) const {
    if (store_by_rows_) {
      return data_[index_row * no_columns_ + index_col];
    }
    return data_[index_col * no_rows_ + index_row];
  }

  Matrix<T>& operator= (const Matrix<T>& src_matrix) {
    no_rows_ = src_matrix.no_rows();
    no_columns_ = src_matrix.no_columns();
    store_by_rows_ = src_matrix.store_by_rows_;
    if (data_ != nullptr) {
      delete[] data_;
    }
    data_ = new T[no_rows_ * no_columns_];
    for (int i = 0; i < no_rows_ * no_columns_; ++i) {
      data_[i] = src_matrix.get_data()[i];
    }

    return *this;
  }

  int no_rows() const { return no_rows_; }
  int no_columns() const { return no_columns_; }

  T* get_data() {
    return data_;
  }

  const T* get_data() const {
    return data_;
  }

 private:
  int no_rows_;
  int no_columns_;
  bool store_by_rows_;
  T* data_;
};

template<int operation>
void ApplyByElement(Matrix<float>* result_matrix,
                    const Matrix<float>& first_matrix,
                    const Matrix<float>& second_matrix);

}  // namespace core
}  // namespace artm


#endif  // SRC_ARTM_CORE_PROCESSOR_H_
