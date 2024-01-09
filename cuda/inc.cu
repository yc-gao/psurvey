extern "C" __global__ void inc(int *nums) { nums[threadIdx.x]++; }
