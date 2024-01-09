#pragma once
#include <functional>

#include <cuda.h>
#include <cuda_runtime.h>

#define NVVM_ASSERT(expr)                                                      \
  do {                                                                         \
    auto err = (expr);                                                         \
    if (err != NVVM_SUCCESS) {                                                 \
      std::cerr << "error libnvvm call failed" << std::endl;                   \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define CU_ASSERT(expr)                                                        \
  do {                                                                         \
    auto err = (expr);                                                         \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *name;                                                        \
      cuGetErrorName(err, &name);                                              \
      const char *msg;                                                         \
      cuGetErrorString(err, &msg);                                             \
      std::cerr << "error " << name << ":" << msg << std::endl;                \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define CUDA_ASSERT(expr)                                                      \
  do {                                                                         \
    auto err = (expr);                                                         \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "error " << cudaGetErrorString(err) << std::endl;           \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

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
