#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cassert>
#include <vector>

#include "cublas_v2.h"
#include "cuda_runtime.h"

#define CUDA_ASSERT(expr)                                                      \
  do {                                                                         \
    auto flag = (expr);                                                        \
    assert(flag == cudaSuccess);                                               \
  } while (0)

#define CUBLAS_ASSERT(expr)                                                    \
  do {                                                                         \
    auto flag = (expr);                                                        \
    assert(flag == CUBLAS_STATUS_SUCCESS);                                     \
  } while (0)

template <typename T, int M, int N, int K> void do_test() {
  class Generator {
    T init_;

  public:
    Generator(T val) : init_(val) {}
    T operator()() { return init_++; }
  };

  std::vector<T> C(M * N);
  std::vector<T> A(M * K);
  std::generate(A.begin(), A.end(), Generator(1));
  std::vector<T> B(K * N);
  std::generate(B.begin(), B.end(), Generator(1));

  T *dev_c;
  T *dev_a;
  T *dev_b;
  CUDA_ASSERT(cudaMalloc(&dev_c, sizeof(T) * M * N));

  CUDA_ASSERT(cudaMalloc(&dev_a, sizeof(T) * M * K));
  CUDA_ASSERT(cudaMalloc(&dev_b, sizeof(T) * K * N));

  cublasHandle_t handle;
  CUBLAS_ASSERT(cublasCreate(&handle));

  CUDA_ASSERT(
      cudaMemcpy(dev_a, A.data(), sizeof(T) * M * K, cudaMemcpyHostToDevice));
  CUDA_ASSERT(
      cudaMemcpy(dev_b, B.data(), sizeof(T) * K * N, cudaMemcpyHostToDevice));

  float alpha = 1, beta = 0;
  for (int i = 0; i < 1024; i++) {
    CUBLAS_ASSERT(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                              dev_b, N, dev_a, K, &beta, dev_c, N));
  }

  CUDA_ASSERT(
      cudaMemcpy(C.data(), dev_c, sizeof(T) * M * N, cudaMemcpyDeviceToHost));
  CUDA_ASSERT(cudaDeviceSynchronize());

  CUBLAS_ASSERT(cublasDestroy(handle));
  CUDA_ASSERT(cudaFree(dev_a));
  CUDA_ASSERT(cudaFree(dev_b));
  CUDA_ASSERT(cudaFree(dev_c));
}

int main(void) {
  do_test<float, 1024, 1024, 1024>();
  return 0;
}
