#include "common.h"

template <typename T, unsigned int warpSize = 32>
__device__ T warpReduce(T val) {
  int laneId = threadIdx.x & 0x1f;
  for (int i = 16; i >= 1; i /= 2)
    val += __shfl_xor_sync(Int2Mask<warpSize>::value, val, i, warpSize);
  return val;
}

template <typename It, typename T>
__global__ void DeviceReduce(It beg, It end, It out) {
  // TODO: impl
}

int main(int argc, char *argv[]) { return 0; }
