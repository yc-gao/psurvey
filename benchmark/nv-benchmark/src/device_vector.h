#pragma once
#include <cuda_runtime.h>
#include <cudnn.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

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

template <typename T>
class device_vector {
  struct cuda_deleter {
    void operator()(T* ptr) { cudaFree(ptr); }
  };

  std::size_t len_;
  std::unique_ptr<T[], cuda_deleter> inner_ptr_;

 public:
  using size_type = std::size_t;
  using value_type = T;
  using pointer_type = T*;
  using const_pointer_type = const T*;

  device_vector() : len_(0) {}

  device_vector(size_type n) : len_(n) {
    pointer_type ptr;
    if (cudaSuccess != cudaMalloc(&ptr, sizeof(T) * len_)) {
      throw std::bad_alloc();
    }
    inner_ptr_.reset(ptr);
  }
  device_vector(size_type n, int val) : device_vector(n) {
    cudaMemset(data(), val, sizeof(T) * size());
  }

  bool operator==(const device_vector& other) const {
    return to_cpu() == other.to_cpu();
  }

  size_type size() const { return len_; }
  pointer_type data() { return inner_ptr_.get(); }
  const_pointer_type data() const { return inner_ptr_.get(); }

  auto to_cpu() const {
    std::vector<T> buf(size());
    cudaMemcpy(buf.data(), data(), sizeof(T) * size(), cudaMemcpyDeviceToHost);
    return buf;
  }
};

template <typename Op, typename... Args>
__global__ void kernel_launcher(Op op, Args&&... args) {
  op.Launch(std::forward<Args>(args)...);
}
