#pragma once
#include <cudnn.h>

#include <iostream>
#include <memory>
#include <stdexcept>

#define COMMON_ASSERT(status, msg)   \
  {                                  \
    auto flag = status;              \
    if (!flag) {                     \
      throw std::runtime_error(msg); \
    }                                \
  }

#define CUDA_ASSERT(status)                                               \
  {                                                                       \
    cudaError_t err = status;                                             \
    if (err != cudaSuccess) {                                             \
      std::stringstream err_msg;                                          \
      err_msg << "CUDA Error: " << cudaGetErrorString(err) << " (" << err \
              << ") at " << __FILE__ << ":" << __LINE__;                  \
      throw std::runtime_error(err_msg.str());                            \
    }                                                                     \
  }

#define CUDNN_ASSERT(status)                                                \
  {                                                                         \
    cudnnStatus_t err = status;                                             \
    if (err != CUDNN_STATUS_SUCCESS) {                                      \
      std::stringstream err_msg;                                            \
      err_msg << "cuDNN Error: " << cudnnGetErrorString(err) << " (" << err \
              << ") at " << __FILE__ << ":" << __LINE__;                    \
      throw std::runtime_error(err_msg.str());                              \
    }                                                                       \
  }

struct CudnnHandleDeleter {
  void operator()(cudnnHandle_t* handle) const {
    if (handle) {
      CUDNN_ASSERT(cudnnDestroy(*handle));
      delete handle;
    }
  }
};

inline std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter>
create_cudnn_handle() {
  auto handle = std::make_unique<cudnnHandle_t>();
  CUDNN_ASSERT(cudnnCreate(handle.get()));
  return std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter>(
      handle.release(), CudnnHandleDeleter());
}
