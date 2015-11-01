// Copyright 2014, Additive Regularization of Topic Models.

#include <artm/utility/blas.h>

#include <boost/filesystem.hpp>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>
#include <utility>

namespace artm {
namespace utility {

namespace {

class Index {
 public:
  Index(int order, int trans, int ld) : order_(order), trans_(trans), ld_(ld) {}
  int operator()(int i, int j) {
    if (trans_ == Blas::Trans) { int tmp = i; i = j; j = tmp; }  // transpose
    return (order_ == Blas::RowMajor) ? (i * ld_ + j) : (i + ld_ * j);
  }

 private:
  int order_;
  int trans_;
  int ld_;
};

float builtin_sdot(int size, const float *x, int xstride, const float *y, int ystride) {
  float result = 0.0f;
  for (int i = 0; i < size; ++i) result += (x[i*xstride] * y[i*ystride]);
  return result;
}

void builtin_saxpy(const int size, const float alpha,
  const float *x, const int xstride,
  float *y, const int ystride) {
  for (int i = 0; i < size; ++i) {
    y[i * ystride] += alpha * x[i * xstride];
  }
}

// Convert sparse matrix from CSC to CSR format
// CSC and CSR format described here: http://docs.nvidia.com/cuda/cusparse/#compressed-sparse-row-format-csr
// Here is short summary for CSR format (Compressed Sparse Row Format):
// - float* _val is an array of length NNZ (represents non-zero values of the matrix),
// - int*   _ptr is an array of length m+1 (represents indices of the first non-zero element in row)
// - int*   _ind is an array of length NNZ (represents column indices of non-zero values of the matrix)
// All indices are 0-based.
void builtin_scsr2csc(int m, int n, int nnz,
  const float *csr_val, const int* csr_row_ptr, const int *csr_col_ind,
  float *csc_val, int* csc_row_ind, int* csc_col_ptr) {
  if (nnz <= 0) return;
  std::vector<std::tuple<int, int, float>> coo_row_ind(nnz);
  for (int i = 0; i < m; ++i) {
    for (int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++) {
      std::get<0>(coo_row_ind[j]) = csr_col_ind[j];
      std::get<1>(coo_row_ind[j]) = i;
      std::get<2>(coo_row_ind[j]) = csr_val[j];
    }
  }

  std::stable_sort(coo_row_ind.begin(), coo_row_ind.end());

  for (int i = 0; i < nnz; ++i) {
    csc_row_ind[i] = std::get<1>(coo_row_ind[i]);
    csc_val[i] = std::get<2>(coo_row_ind[i]);
  }

  csc_col_ptr[n] = nnz;
  for (int j = 0, i = 0; j < n; ++j) {
    csc_col_ptr[j] = i;
    while ((i < nnz) && (std::get<0>(coo_row_ind[i]) == j))
      i++;
  }
}


void builtin_sgemm(int order, const int transa, const int transb,
  const int m, const int n, const int k,
  const float alpha,
  const float * a, const int lda,
  const float * b, const int ldb,
  const float beta,
  float * c, const int ldc) {
  Index ia(order, transa, lda);
  Index ib(order, transb, ldb);
  Index ic(order, Blas::NoTrans, ldc);

  bool rowa_contiguous = (order == Blas::ColMajor) ? (transa == Blas::Trans) : (transa != Blas::Trans);
  bool colb_contiguous = (order == Blas::ColMajor) ? (transb != Blas::Trans) : (transb == Blas::Trans);

  // Remember that if any stride is non-contiguous then computation will be ~10 times slower.
  // In such case consider to store transposed version of the matrix.
  int astride = rowa_contiguous ? 1 : lda;
  int bstride = colb_contiguous ? 1 : ldb;

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      const float* aa = a + ia(i, 0);
      const float* bb = b + ib(0, j);
      float& cc = c[ic(i, j)];
      float result = builtin_sdot(k, aa, astride, bb, bstride);
      cc = alpha * result + cc * beta;
    }
  }
}

class BuiltinBlas : public Blas {
 public:
  BuiltinBlas() {
    sgemm = builtin_sgemm;
    sdot = builtin_sdot;
    saxpy = builtin_saxpy;
    scsr2csc = builtin_scsr2csc;
  }

  virtual bool is_loaded() { return true; }
};

}  // namespace

std::once_flag flag_builtin;

Blas* Blas::builtin() {
  static BuiltinBlas* impl;
  std::call_once(flag_builtin, [](){ impl = new BuiltinBlas(); });
  return impl;
}

}  // namespace utility
}  // namespace artm
