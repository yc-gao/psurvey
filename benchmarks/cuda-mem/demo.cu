#include <cassert>
#include <iostream>
#include <string>

__global__ void do_copy0(int *dst, const int *src, int l) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < l;
       idx += blockDim.x * gridDim.x) {
    dst[idx] = src[idx];
  }
}

__global__ void do_copy1(int *dst, const int *src, int l) {
  auto itemsPerThread =
      (l + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
  for (int idx = (threadIdx.x + blockIdx.x * blockDim.x) * itemsPerThread;
       idx < l; idx++) {
    dst[idx] = src[idx];
  }
}

template <typename F>
void do_test(std::string name, F &&f) {
  cudaEvent_t st, ed;
  assert(cudaSuccess == cudaEventCreate(&st));
  assert(cudaSuccess == cudaEventCreate(&ed));
  f();

  cudaEventRecord(st);
  for (int i = 0; i < 100; i++) {
    f();
  }
  assert(cudaSuccess == cudaEventRecord(ed));
  assert(cudaSuccess == cudaEventSynchronize(ed));
  float tm = 0;
  assert(cudaSuccess == cudaEventElapsedTime(&tm, st, ed));
  std::cout << name << " time " << tm << " ms" << std::endl;
}

int main(int argc, char *argv[]) {
  int l = 1024 * 1024 * 1024;
  int *d_a, *d_b;
  assert(cudaSuccess == cudaMalloc(&d_a, sizeof(int) * l));
  assert(cudaSuccess == cudaMalloc(&d_b, sizeof(int) * l));

  do_test("copy0", [=]() { do_copy0<<<1024, 1024>>>(d_a, d_b, l); });
  do_test("copy1", [=]() { do_copy0<<<1024, 1024>>>(d_a, d_b, l); });
  do_test("cudaMemcpy", [=]() {
    assert(cudaSuccess ==
           cudaMemcpy(d_a, d_b, sizeof(int) * l, cudaMemcpyDeviceToDevice));
  });
  return 0;
}
