#include <cblas.h>
#include <stdio.h>

void row_major() {
  int i = 0;
  double A[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};     // A(3x2)
  double B[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};     // B(2x3)
  double C[9] = {.5, .5, .5, .5, .5, .5, .5, .5, .5}; // C(3x3)

  const int M = 3; // row of A and C
  const int N = 3; // col of B and C
  const int K = 2; // col of A and row of B

  const double alpha = 1.0;
  const double beta = 0.1;

  // C = alpha * A * B + beta * C
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K,
              B, N, beta, C, N);

  for (i = 0; i < 9; i++) {
    printf("%lf ", C[i]);
  }
  printf("\n");
}

void col_major() {
  int i = 0;
  double A[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};     // A(3x2)
  double B[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};     // B(2x3)
  double C[9] = {.5, .5, .5, .5, .5, .5, .5, .5, .5}; // C(3x3)

  const int M = 3; // row of A and C
  const int N = 3; // col of B and C
  const int K = 2; // col of A and row of B

  const double alpha = 1.0;
  const double beta = 0.1;

  // C = alpha * A * B + beta * C
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, B, N,
              A, K, beta, C, N);

  for (i = 0; i < 9; i++) {
    printf("%lf ", C[i]);
  }
  printf("\n");
}

void trans_col_major() {
  int i = 0;
  double A[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};     // A(3x2)
  double B[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};     // B(2x3)
  double C[9] = {.5, .5, .5, .5, .5, .5, .5, .5, .5}; // C(3x3)

  const int M = 3; // row of A and C
  const int N = 3; // col of B and C
  const int K = 2; // col of A and row of B

  const double alpha = 1.0;
  const double beta = 0.1;

  // C = alpha * A * B + beta * C
  cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, M, N, K, alpha, B, K, A, M,
              beta, C, N);

  for (i = 0; i < 9; i++) {
    printf("%lf ", C[i]);
  }
  printf("\n");
}

int main() {
  row_major();
  col_major();
  trans_col_major();
  return 0;
}
