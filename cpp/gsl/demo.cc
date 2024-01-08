#include <iostream>
#include <stdio.h>

#include "gsl/gsl_blas.h"
#include "gsl/gsl_poly.h"

void test_blas() {
  double a[] = {0.11, 0.12, 0.13, 0.21, 0.22, 0.23};

  double b[] = {1011, 1012, 1021, 1022, 1031, 1032};

  double c[] = {0.00, 0.00, 0.00, 0.00};

  gsl_matrix_view A = gsl_matrix_view_array(a, 2, 3);
  gsl_matrix_view B = gsl_matrix_view_array(b, 3, 2);
  gsl_matrix_view C = gsl_matrix_view_array(c, 2, 2);

  /* Compute C = A B */

  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &A.matrix, &B.matrix, 0.0,
                 &C.matrix);

  printf("[ %g, %g\n", c[0], c[1]);
  printf("  %g, %g ]\n", c[2], c[3]);
}

void test_poly() {
  double c[] = {1, 1, 1, 1};
  auto r = gsl_poly_eval(c, sizeof(c) / sizeof(c[0]), 1);
  std::cout << "poly result " << r << std::endl;
}

int main(int argc, char *argv[]) {
  test_poly();
  test_blas();
  return 0;
}
