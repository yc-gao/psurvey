#include <iostream>

#include "common.h"

__global__ void inc(float *num) { *num += 1; }

int main(int argc, char *argv[]) {
  cudaDeviceProp prop;
  nv_assert(cudaGetDeviceProperties(&prop, 0));
  if (!prop.canMapHostMemory) {
    std::cout << "can not map host memory" << std::endl;
    return 0;
  }
  nv_assert(cudaSetDeviceFlags(cudaDeviceMapHost));

  float *a_h, *a_map;
  nv_assert(cudaHostAlloc(&a_h, sizeof(float), cudaHostAllocMapped));
  *a_h = 1;
  nv_assert(cudaHostGetDevicePointer(&a_map, a_h, 0));

  inc<<<1, 1>>>(a_map);
  nv_assert(cudaDeviceSynchronize());

  std::cout << *a_h << std::endl;
  return 0;
}
