// Copyright 2014, Additive Regularization of Topic Models.

#include "gtest/gtest.h"

#include "artm/utility/blas.h"

using namespace artm::utility;  // NOLINT

// To run this particular test:
// artm_tests.exe --gtest_filter=Blas.*
TEST(Blas, Basic) {
  float A[6] = { 1, 2, 3, 4, 5, 6 };
  float A2[6] = { 0, 0, 0, 0, 0, 0 };
  float B[8] = { 7, 8, 9, 10, 11, 12, 13, 14 };
  float AT[6] = { 1, 3, 5, 2, 4, 6 };
  float BT[8] = { 7, 11, 8, 12, 9, 13, 10, 14 };
  float C1[12] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  float C[12] = { 29, 32, 35, 38, 65, 72, 79, 86, 101, 112, 123, 134 };
  float CT[12] = { 29, 65, 101, 32, 72, 112, 35, 79, 123, 38, 86, 134 };

  Blas& blas_builtin = Blas::builtin();
  Blas& blas_mkl = Blas::mkl();
  ASSERT_TRUE(blas_builtin.is_loaded());
  if (!blas_mkl.is_loaded()) {
    LOG(WARNING) << "Intel Math Kernel Library not loaded";
  }

  Blas* blas[2] = { &blas_builtin, &blas_mkl };
  for (int index = 0; index < 2; ++index) {
    if (!blas_mkl.is_loaded()) continue;
    EXPECT_EQ(blas[index]->sdot(12, C, 1, CT, 1), 75188);
    blas[index]->saxpy(6, 1.0, A, 1, A2, 1);
    for (int i = 0; i < 6; ++i) EXPECT_EQ(A2[i], A[i]);
    blas[index]->saxpy(6, -1.0, A, 1, A2, 1);
    blas[index]->sgemm(Blas::RowMajor, Blas::NoTrans, Blas::NoTrans, 3, 4, 2, 1.0, A, 2, B, 4, 0, C1, 4);
    for (int i = 0; i < 12; ++i) EXPECT_EQ(C1[i], C[i]);
    blas[index]->sgemm(Blas::ColMajor, Blas::NoTrans, Blas::NoTrans, 3, 4, 2, 1.0, AT, 3, BT, 2, 0, C1, 3);
    for (int i = 0; i < 12; ++i) EXPECT_EQ(C1[i], CT[i]);
    blas[index]->sgemm(Blas::RowMajor, Blas::Trans, Blas::NoTrans, 3, 4, 2, 1.0, AT, 3, B, 4, 0, C1, 4);
    for (int i = 0; i < 12; ++i) EXPECT_EQ(C1[i], C[i]);
  }
}

TEST(Blas, scsr2csc) {
  int m = 4, n = 5, nnz = 8;
  float csr_val2[8], csr_val[8] = { 10, 11, 12, 13, 14, 15, 16, 17 };
  int csr_row_ptr2[5], csr_row_ptr[5] = { 0, 3, 3, 6, 8 };
  int csr_col_ind2[8], csr_col_ind[8] = { 0, 2, 4, 1, 2, 4, 0, 4 };
  float csc_val[8], csc_val_exp[8] = { 10, 16, 13, 11, 14, 12, 15, 17 };
  int csc_col_ptr[6], csc_col_ptr_exp[6] = { 0, 2, 3, 5, 5, 8 };
  int csc_row_ind[8], csc_row_ind_exp[8] = { 0, 3, 2, 0, 2, 0, 2, 3 };
  Blas& blas = Blas::builtin();
  ASSERT_TRUE(blas.is_loaded());
  blas.scsr2csc(m, n, nnz, csr_val, csr_row_ptr, csr_col_ind, csc_val, csc_row_ind, csc_col_ptr);
  for (int i = 0; i < nnz; ++i) EXPECT_EQ(csc_val[i], csc_val_exp[i]);
  for (int i = 0; i < nnz; ++i) EXPECT_EQ(csc_row_ind[i], csc_row_ind_exp[i]);
  for (int i = 0; i < (n + 1); ++i) EXPECT_EQ(csc_col_ptr[i], csc_col_ptr_exp[i]);

  // convert back
  blas.scsr2csc(n, m, nnz, csc_val, csc_col_ptr, csc_row_ind, csr_val2, csr_col_ind2, csr_row_ptr2);
  for (int i = 0; i < nnz; ++i) EXPECT_EQ(csr_val2[i], csr_val[i]);
  for (int i = 0; i < nnz; ++i) EXPECT_EQ(csr_col_ind2[i], csr_col_ind[i]);
  for (int i = 0; i < (m + 1); ++i) EXPECT_EQ(csr_row_ptr2[i], csr_row_ptr[i]);
}
