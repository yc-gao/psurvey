#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

#include <cstdint>

__global__ void ldmatrix_kernel() {
  __shared__ half smem[2 * 8 * 8];
  if (threadIdx.x == 0) {
    for (auto i = 0; i < 2 * 8 * 8; i++) {
      smem[i] = __int2half_rd(i);
    }
  }
  __syncthreads();
  std::uint32_t smem_ptr = __cvta_generic_to_shared(smem + threadIdx.x * 8);
  std::uint32_t dst0, dst1;
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(dst0), "=r"(dst1)
               : "r"(smem_ptr));

  printf("thread %d val %f:%f %f:%f\n", threadIdx.x,
         __half2float(reinterpret_cast<half *>(&dst0)[0]),
         __half2float(reinterpret_cast<half *>(&dst0)[1]),
         __half2float(reinterpret_cast<half *>(&dst1)[0]),
         __half2float(reinterpret_cast<half *>(&dst1)[1]));
}
void ldmatrix_demo() {
  ldmatrix_kernel<<<1, 32>>>();
  cudaDeviceSynchronize();
}

int main(int argc, char *argv[]) {
  ldmatrix_demo();
  return 0;
}
