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

class Blas_interface {
 public:
  virtual ~Blas_interface() {}
  virtual bool is_loaded() = 0;
  blas_sgemm_type* sgemm;
  blas_saxpy_type* saxpy;
  blas_sdot_type*  sdot;
  blas_scsr2csc_type* scsr2csc;
};

class Blas : boost::noncopyable, public Blas_interface {
 public:
  enum LibraryType {
    BUILTIN,
    MKL
  };

  static const int RowMajor = 101;
  static const int ColMajor = 102;
  static const int NoTrans = 111;
  static const int Trans = 112;
  static const int ConfTrans = 113;

  static Blas& singleton(LibraryType library_type) {
    static Blas blas_builtin(BUILTIN);
    static Blas blas_mkl(MKL);

    switch (library_type) {
      case BUILTIN:
        return blas_builtin;
      case MKL:
        return blas_mkl;
    }

    throw std::runtime_error("Invalid library type");
  }

  virtual bool is_loaded() { return (impl_ != nullptr) && impl_->is_loaded(); }

 private:
  explicit Blas(LibraryType library_type);  // Singleton (make constructor private)
  std::unique_ptr<Blas_interface> impl_;
};


}  // namespace utility
}  // namespace artm

#endif  // SRC_ARTM_UTILITY_BLAS_H_
