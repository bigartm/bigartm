// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_UTILITY_BLAS_H_
#define SRC_ARTM_UTILITY_BLAS_H_

#include <memory>
#include "boost/utility.hpp"
#include "glog/logging.h"

#include "artm/utility/ice.h"

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

  static Blas& mkl();
  static Blas& builtin();

 protected:
  Blas() {};  // Singleton (make constructor private)
};


}  // namespace utility
}  // namespace artm

#endif  // SRC_ARTM_UTILITY_BLAS_H_
