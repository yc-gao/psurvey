#include <iostream>

#include <nvrtc.h>

#include "common.h"

const std::string kernel_source = R"(
extern "C" __global__ void inc(int *nums) { nums[threadIdx.x]++; }
)";

std::string GetLog(nvrtcProgram &prog) {
  size_t logSizeRet;
  nv_assert(nvrtcGetProgramLogSize(prog, &logSizeRet));
  std::string log;
  log.resize(logSizeRet);
  nv_assert(nvrtcGetProgramLog(prog, log.data()));
  return log;
}

std::string Kernel2Ptx(const std::string &kernel_source,
                       const std::string &name) {
  nvrtcProgram prog;
  nv_assert(nvrtcCreateProgram(&prog, kernel_source.c_str(), name.c_str(), 0,
                               NULL, NULL));
  MAKE_DEFER(nv_assert(nvrtcDestroyProgram(&prog)));

  const char *opts[] = {"--gpu-architecture=compute_80", "--fmad=false"};
  nv_assert(nvrtcCompileProgram(prog, sizeof(opts) / sizeof(opts[0]), opts));

  size_t ptxSize;
  nv_assert(nvrtcGetPTXSize(prog, &ptxSize));
  std::string ptx;
  ptx.resize(ptxSize);
  nv_assert(nvrtcGetPTX(prog, ptx.data()));
  return ptx;
}

// TODO: kernel to cubin
// std::string Kernel2Cubin(const std::string &kernel_source,
//                          const std::string &name) {
//   nvrtcProgram prog;
//   nv_assert(nvrtcCreateProgram(&prog, kernel_source.c_str(), name.c_str(), 0,
//                                NULL, NULL));
//   MAKE_DEFER(nv_assert(nvrtcDestroyProgram(&prog)));
//
//   const char *opts[] = {"--gpu-architecture=compute_80", "--fmad=false"};
//   nv_assert(nvrtcCompileProgram(prog, sizeof(opts) / sizeof(opts[0]), opts));
//
//   size_t cubinSizeRet;
//   nv_assert(nvrtcGetCUBINSize(prog, &cubinSizeRet));
//   std::string cubin;
//   cubin.resize(cubinSizeRet);
//   nv_assert(nvrtcGetCUBIN(prog, cubin.data()));
//   return cubin;
// }
//
// TODO: kernel to nvvm
// std::string Kernel2NVVM(const std::string &kernel_source,
//                         const std::string &name) {
//   nvrtcProgram prog;
//   nv_assert(nvrtcCreateProgram(&prog, kernel_source.c_str(), name.c_str(), 0,
//                                NULL, NULL));
//   MAKE_DEFER(nv_assert(nvrtcDestroyProgram(&prog)));
//
//   const char *opts[] = {};
//   nv_assert(nvrtcCompileProgram(prog, sizeof(opts) / sizeof(opts[0]), opts));
//
//   size_t nvvmSizeRet;
//   nv_assert(nvrtcGetNVVMSize(prog, &nvvmSizeRet));
//   std::string nvvm;
//   nvvm.resize(nvvmSizeRet);
//   nv_assert(nvrtcGetNVVM(prog, nvvm.data()));
//   return nvvm;
// }

int main(int argc, char *argv[]) {
  std::cout << "ptx:\n" << Kernel2Ptx(kernel_source, "inc.cu") << std::endl;
  // std::string cubin = Kernel2Cubin(kernel_source, "inc.cu");
  // std::cout << "cubin size " << cubin.size() << std::endl;
  // std::cout << "nvvm:\n" << Kernel2NVVM(kernel_source, "inc.cu") <<
  // std::endl;
  return 0;
}
