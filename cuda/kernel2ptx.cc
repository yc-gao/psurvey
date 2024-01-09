#include <iostream>

#include <nvrtc.h>

#include "common.h"

const std::string kernel_source = R"(
extern "C" __global__ void inc(int *nums) { nums[threadIdx.x]++; }
)";

int main(int argc, char *argv[]) {
  nvrtcProgram prog;
  nv_assert(nvrtcCreateProgram(&prog, kernel_source.c_str(), "inc.cu", 0, NULL,
                               NULL));

  const char *opts[] = {"--fmad=false"};
  nv_assert(nvrtcCompileProgram(prog, 1, opts));

  size_t ptxSize;
  char *ptx;
  nv_assert(nvrtcGetPTXSize(prog, &ptxSize));
  ptx = (char *)malloc(ptxSize);
  nv_assert(nvrtcGetPTX(prog, ptx));
  std::cout << std::string(ptx, ptxSize) << std::endl;

  return 0;
}
