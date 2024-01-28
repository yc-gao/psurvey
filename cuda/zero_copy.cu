#include <iostream>

#include "common.h"

__global__ void inc(float *num, int n) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto idx = tid;
  while (idx < n) {
    num[idx] += 1;
    idx += gridDim.x * blockDim.x;
  }
}

template <std::size_t N> void do_zerocopy() {
  cudaDeviceProp prop;
  nv_assert(cudaGetDeviceProperties(&prop, 0));
  if (!prop.canMapHostMemory) {
    std::cout << "can not map host memory" << std::endl;
    return;
  }
  nv_assert(cudaSetDeviceFlags(cudaDeviceMapHost));

  float *a_h, *a_map;
  nv_assert(cudaHostAlloc(&a_h, sizeof(float) * N, cudaHostAllocMapped));
  MAKE_DEFER(cudaFreeHost(a_h));
  nv_assert(cudaHostGetDevicePointer(&a_map, a_h, 0));
  std::fill_n(a_h, N, 0);

  cudaEvent_t st, ed;
  nv_assert(cudaEventCreate(&st));
  MAKE_DEFER(cudaEventDestroy(st));
  nv_assert(cudaEventCreate(&ed));
  MAKE_DEFER(cudaEventDestroy(ed));

  inc<<<1, 1>>>(a_map, N);
  nv_assert(cudaDeviceSynchronize());
  inc<<<1, 1>>>(a_map, N);
  nv_assert(cudaEventRecord(st, 0));
  for (int i = 0; i < 100; i++) {
    inc<<<1, 1>>>(a_map, N);
  }
  nv_assert(cudaEventRecord(ed, 0));
  nv_assert(cudaDeviceSynchronize());
  float tm;
  nv_assert(cudaEventElapsedTime(&tm, st, ed));
  std::cout << "zerocopy time " << tm << "ms" << std::endl;
}

template <std::size_t N> void do_normal() {
  float a_h[N];
  std::fill_n(a_h, N, 0);
  float *a_map;
  nv_assert(cudaMalloc(&a_map, sizeof(a_h)));
  MAKE_DEFER(cudaFree(a_map));
  nv_assert(cudaMemcpy(a_map, a_h, sizeof(a_h), cudaMemcpyHostToDevice));

  cudaEvent_t st, ed;
  nv_assert(cudaEventCreate(&st));
  MAKE_DEFER(cudaEventDestroy(st));
  nv_assert(cudaEventCreate(&ed));
  MAKE_DEFER(cudaEventDestroy(ed));

  inc<<<1, 1>>>(a_map, N);
  nv_assert(cudaDeviceSynchronize());
  inc<<<1, 1>>>(a_map, N);
  nv_assert(cudaEventRecord(st, 0));
  for (int i = 0; i < 100; i++) {
    inc<<<1, 1>>>(a_map, N);
  }
  nv_assert(cudaEventRecord(ed, 0));
  nv_assert(cudaDeviceSynchronize());
  float tm;
  nv_assert(cudaEventElapsedTime(&tm, st, ed));
  std::cout << "normal time " << tm << "ms" << std::endl;
}

int main(int argc, char *argv[]) {
  do_zerocopy<4096>();
  do_normal<4096>();
  return 0;
}
