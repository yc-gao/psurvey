#include <vector>

#include "common.h"

__global__ void inc(int *nums, int l) {
  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < l;
       idx += gridDim.x * blockDim.x) {
    nums[idx]++;
  }
}

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  std::vector<int> nums(1024);
  int *d_nums;
  nv_assert(cudaMalloc(&d_nums, sizeof(int) * nums.size()));
  nv_assert(cudaMemcpy(d_nums, nums.data(), sizeof(int) * nums.size(),
                       cudaMemcpyHostToDevice));
  inc<<<256, 256>>>(d_nums, nums.size());
  nv_assert(cudaMemcpy(nums.data(), d_nums, sizeof(int) * nums.size(),
                       cudaMemcpyDeviceToHost));
  nv_assert(cudaDeviceSynchronize());
  nv_assert(cudaFree(d_nums));
  return 0;
}
