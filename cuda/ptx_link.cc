#include <string.h>

#include <algorithm>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"

const std::string ptx_source = R"(
// nvcc -o inc.ptx --ptx inc.cu
.version 7.8
.target sm_52

.visible .entry inc(
	.param .u64 inc_param_0
)
{
	.reg .b32 	%r<4>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [inc_param_0];
	cvta.to.global.u64 	%rd2, %rd1;
	mov.u32 	%r1, %tid.x;
	mul.wide.u32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.u32 	%r2, [%rd4];
	add.s32 	%r3, %r2, 1;
	st.global.u32 	[%rd4], %r3;
	ret;

}

)";

int main(int argc, char *argv[]) {
  int driverVersion;
  CU_ASSERT(cuDriverGetVersion(&driverVersion));
  std::cout << "cu init, driver version " << driverVersion << std::endl;

  CU_ASSERT(cuInit(0));

  int device_count = -1;
  CU_ASSERT(cuDeviceGetCount(&device_count));
  std::cout << "device count " << device_count << std::endl;
  if (device_count <= 0) {
    return 0;
  }

  CUdevice device;
  CU_ASSERT(cuDeviceGet(&device, 0));
  std::cout << "get cuda device idx 0" << std::endl;

  // init device context
  cudaFree(0);

  CUlinkState lState;

  CUjit_option options[] = {};
  void *optionVals[] = {};
  CU_ASSERT(cuLinkCreate(sizeof(options) / sizeof(options[0]), options,
                         optionVals, &lState));
  CU_ASSERT(cuLinkAddData(lState, CU_JIT_INPUT_PTX, (void *)ptx_source.c_str(),
                          strlen(ptx_source.c_str()) + 1, 0, 0, 0, 0));
  void *cuOut;
  size_t outSize;
  CU_ASSERT(cuLinkComplete(lState, &cuOut, &outSize));
  std::cout << "jit module size " << outSize << std::endl;
  MAKE_DEFER(CU_ASSERT(cuLinkDestroy(lState)));

  CUmodule hModule;
  CU_ASSERT(cuModuleLoadData(&hModule, cuOut));
  MAKE_DEFER(CU_ASSERT(cuModuleUnload(hModule)));

  CUfunction hKernel = 0;
  CU_ASSERT(cuModuleGetFunction(&hKernel, hModule, "inc"));

  {
    int nums[32];
    std::generate(std::begin(nums), std::end(nums), []() {
      static int n = 0;
      return n++;
    });
    std::cout << "init data:";
    for (auto beg = std::begin(nums), end = std::end(nums); beg != end; beg++) {
      std::cout << " " << *beg;
    }
    std::cout << std::endl;
    int *d_nums;
    CUDA_ASSERT(cudaMalloc(&d_nums, sizeof(nums)));
    MAKE_DEFER(cudaFree(d_nums));
    CUDA_ASSERT(cudaMemcpy(d_nums, nums, sizeof(nums), cudaMemcpyHostToDevice));

    void *args[] = {&d_nums};
    CU_ASSERT(cuLaunchKernel(hKernel, 1, 1, 1, 32, 1, 1, 0, 0, args, 0));
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaMemcpy(nums, d_nums, sizeof(nums), cudaMemcpyDeviceToHost));
    std::cout << "result data:";
    for (auto beg = std::begin(nums), end = std::end(nums); beg != end; beg++) {
      std::cout << " " << *beg;
    }
    std::cout << std::endl;
  }
  return 0;
}
