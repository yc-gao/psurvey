#pragma once
#include <functional>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

inline void nv_assert(nvrtcResult err) {
  if (err != NVRTC_SUCCESS) {
    std::cerr << "error nvrtc call failed" << std::endl;
    exit(-1);
  }
}

inline void nv_assert(cudaError_enum err) {
  if (err != CUDA_SUCCESS) {
    const char *name;
    cuGetErrorName(err, &name);
    const char *msg;
    cuGetErrorString(err, &msg);
    std::cerr << "error " << name << ":" << msg << std::endl;
    exit(-1);
  }
}

inline void nv_assert(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "error " << cudaGetErrorString(err) << std::endl;
    exit(-1);
  }
}

class Defer {
  std::function<void()> cb_;

public:
  Defer(const Defer &) = default;
  Defer(Defer &&) = default;
  Defer &operator=(const Defer &) = default;
  Defer &operator=(Defer &&) = default;

  ~Defer() {
    if (cb_) {
      cb_();
    }
  }
  template <typename F> Defer(F &&cb) : cb_(std::forward<F>(cb)) {}
};
#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)
#define MAKE_DEFER(func) Defer CONCATENATE(defer__, __LINE__)([&]() { func; })

template <unsigned int N, typename = int> struct Int2Mask;
template <unsigned int N, typename T> struct Int2Mask {
  constexpr static T value = (Int2Mask<N - 1, T>::value << 1) | 1;
};
template <typename T> struct Int2Mask<0, T> {
  constexpr static T value = 0;
};
