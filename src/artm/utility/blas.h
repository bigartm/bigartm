// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_UTILITY_BLAS_H_
#define SRC_ARTM_UTILITY_BLAS_H_

#include <assert.h>
#include <memory>
#include <vector>

#include "boost/exception/diagnostic_information.hpp"
#include "boost/utility.hpp"

#include "glog/logging.h"

typedef void blas_sgemm_type(int order, const int transa, const int transb,
                             const int m, const int n, const int k,
                             const float alpha,
                             const float * a, const int lda,
                             const float * b, const int ldb,
                             const float beta,
                             float * c, const int ldc);

typedef float blas_sdot_type(int size,
                             const float *x, int xstride,
                             const float *y, int ystride);

typedef void blas_saxpy_type(const int size, const float alpha,
                             const float *x, const int xstride,
                             float *y, const int ystride);

typedef void blas_scsr2csc_type(int m, int n, int nnz,
                            const float *csr_val, const int* csr_row_ptr, const int *csr_col_ind,
                                  float *csc_val,       int* csc_row_ind,       int* csc_col_ptr);

#define CATCH_BIG_ALLOCATION(no_rows, no_cols)                                      \
catch (...) {                                                                       \
  LOG(ERROR) << "no_rows_ = " << no_rows << ", no_columns_ = " << no_cols << ". "   \
    << boost::current_exception_diagnostic_information();                           \
  throw;                                                                            \
}

namespace artm {
namespace utility {

class Blas {
 public:
  virtual ~Blas() {}
  virtual bool is_loaded() = 0;
  blas_sgemm_type* sgemm;
  blas_saxpy_type* saxpy;
  blas_sdot_type*  sdot;
  blas_scsr2csc_type* scsr2csc;

  static const int RowMajor = 101;
  static const int ColMajor = 102;
  static const int NoTrans = 111;
  static const int Trans = 112;
  static const int ConfTrans = 113;

  static Blas* builtin();

 protected:
  Blas() { }  // Singleton (make constructor private)
};

template<typename T>
class DenseMatrix {
 public:
  DenseMatrix(int no_rows = 0, int no_columns = 0, bool store_by_rows = true)
    : no_rows_(no_rows),
    no_columns_(no_columns),
    store_by_rows_(store_by_rows),
    data_(nullptr) {
    if (no_rows > 0 && no_columns > 0) {
      try {
        data_ = new T[no_rows_ * no_columns_];
      } CATCH_BIG_ALLOCATION(no_rows_, no_columns_)
    }
  }

  DenseMatrix(const DenseMatrix<T>& src_matrix) {
    no_rows_ = src_matrix.no_rows();
    no_columns_ = src_matrix.no_columns();
    store_by_rows_ = src_matrix.store_by_rows_;
    if (no_columns_ >0 && no_rows_ > 0) {
      try {
        data_ = new T[no_rows_ * no_columns_];
      } CATCH_BIG_ALLOCATION(no_rows_, no_columns_)

      for (int i = 0; i < no_rows_ * no_columns_; ++i) {
        data_[i] = src_matrix.get_data()[i];
      }
    } else {
      data_ = nullptr;
    }
  }

  ~DenseMatrix() {
    delete[] data_;
  }

  void InitializeZeros() {
    memset(data_, 0, sizeof(T)* no_rows_ * no_columns_);
  }

  T& operator() (int index_row, int index_col) {
    assert(index_row < no_rows_);
    assert(index_col < no_columns_);
    if (store_by_rows_) {
      return data_[index_row * no_columns_ + index_col];
    }
    return data_[index_col * no_rows_ + index_row];
  }

  const T& operator() (int index_row, int index_col) const {
    assert(index_row < no_rows_);
    assert(index_col < no_columns_);
    if (store_by_rows_) {
      return data_[index_row * no_columns_ + index_col];
    }
    return data_[index_col * no_rows_ + index_row];
  }

  DenseMatrix<T>& operator= (const DenseMatrix<T>& src_matrix) {
    no_rows_ = src_matrix.no_rows();
    no_columns_ = src_matrix.no_columns();
    store_by_rows_ = src_matrix.store_by_rows_;
    if (data_ != nullptr) {
      delete[] data_;
    }
    if (no_columns_ >0 && no_rows_ > 0) {
      try {
        data_ = new  T[no_rows_ * no_columns_];
      } CATCH_BIG_ALLOCATION(no_rows_, no_columns_);

      for (int i = 0; i < no_rows_ * no_columns_; ++i) {
        data_[i] = src_matrix.get_data()[i];
      }
    } else {
      data_ = nullptr;
    }

    return *this;
  }

  int no_rows() const { return no_rows_; }
  int no_columns() const { return no_columns_; }
  int size() const { return no_rows_ * no_columns_; }
  bool is_equal_size(const DenseMatrix<T>& rhs) {
    return no_rows_ == rhs.no_rows_ && no_columns_ == rhs.no_columns_;
  }

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

template<typename T>
class CsrMatrix {
 public:
  explicit CsrMatrix(int m, int n, int nnz) : m_(m), n_(n), nnz_(nnz) {
    assert(m > 0 && n > 0 && nnz > 0);
    val_.resize(nnz);
    col_ind_.resize(nnz);
    row_ptr_.resize(m + 1);
  }

  explicit CsrMatrix(int n, std::vector<T>* val, std::vector<int>* row_ptr, std::vector<int>* col_ind) {
    assert(val != nullptr && row_ptr != nullptr && col_ind != nullptr);
    m_ = static_cast<int>(row_ptr->size()) - 1;
    n_ = n;  // this parameter can't be deduced automatically
    nnz_ = static_cast<int>(val->size());
    val_.swap(*val);
    row_ptr_.swap(*row_ptr);
    col_ind_.swap(*col_ind);
  }

  void Transpose(artm::utility::Blas* blas) {
    std::vector<int> row_ptr_new_(n_ + 1);
    blas->scsr2csc(m_, n_, nnz_, val(), row_ptr(), col_ind(), val(), col_ind(), &row_ptr_new_[0]);
    int tmp = m_; m_ = n_; n_ = tmp;  // swat(m, n)
    row_ptr_.swap(row_ptr_new_);
  }

  T* val() { return &val_[0]; }
  const T* val() const { return &val_[0]; }

  int* row_ptr() { return &row_ptr_[0]; }
  const int* row_ptr() const { return &row_ptr_[0]; }

  int* col_ind() { return &col_ind_[0]; }
  const int* col_ind() const { return &col_ind_[0]; }

  int m() const { return m_; }
  int n() const { return n_; }
  int nnz() const { return nnz_; }

 private:
  int m_;
  int n_;
  int nnz_;
  std::vector<T> val_;
  std::vector<int> row_ptr_;
  std::vector<int> col_ind_;
};

template<typename T>
void AssignDenseMatrixByProduct(const DenseMatrix<T>& first_matrix,
                                const DenseMatrix<T>& second_matrix,
                                DenseMatrix<T>* result_matrix) {
  assert(result_matrix->is_equal_size(first_matrix));
  assert(result_matrix->is_equal_size(second_matrix));

  T* result_data = result_matrix->get_data();
  const T* first_data = first_matrix.get_data();
  const T* second_data = second_matrix.get_data();
  const int size = result_matrix->size();

  for (int i = 0; i < size; ++i)
    result_data[i] = first_data[i] * second_data[i];
}

template<typename T>
void AssignDenseMatrixByDivision(const DenseMatrix<T>& first_matrix,
                                 const DenseMatrix<T>& second_matrix,
                                 DenseMatrix<T>* result_matrix) {
  assert(result_matrix->is_equal_size(first_matrix));
  assert(result_matrix->is_equal_size(second_matrix));

  T* result_data = result_matrix->get_data();
  const T* first_data = first_matrix.get_data();
  const T* second_data = second_matrix.get_data();
  const int size = result_matrix->size();

  for (int i = 0; i < size; ++i) {
    if (first_data[i] == 0 || second_data[i] == 0)
      result_data[i] = 0;
    else
      result_data[i] = first_data[i] / second_data[i];
  }
}

}  // namespace utility
}  // namespace artm

#undef CATCH_BIG_ALLOCATION

#endif  // SRC_ARTM_UTILITY_BLAS_H_
