#pragma once
#include <cuda_runtime.h>

#include <stdexcept>

inline void nv_assert(cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error{std::string("cuda assert failed, msg: ") +
                             cudaGetErrorString(err)};
  }
}
