#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <new>
#include <vector>

template <typename T>
class device_vector {
  struct cuda_deleter {
    void operator()(T* ptr) { cudaFree(ptr); }
  };

 public:
  using size_type = std::size_t;
  using value_type = T;
  using pointer_type = T*;
  using const_pointer_type = T const*;

  device_vector() = default;
  device_vector(size_type len) : len_(len) {
    pointer_type ptr{};
    if (cudaSuccess != cudaMalloc(&ptr, sizeof(T) * len_)) {
      throw std::bad_alloc();
    }
    ptr_.reset(ptr);
  }

  size_type size() const { return len_; }

  pointer_type data() { return ptr_.get(); }
  auto begin() { return data(); }
  auto end() { return data() + size(); }

  const_pointer_type data() const { return ptr_.get(); }
  auto begin() const { return data(); }
  auto end() const { return data() + size(); }

  device_vector<T>& From(const std::vector<T>& other) {
    if (cudaSuccess != cudaMemcpy(data(), other.data(), size() * sizeof(T),
                                  cudaMemcpyHostToDevice)) {
      throw std::runtime_error("can't do host2device copy");
    }
    return *this;
  }

  void To(std::vector<T>& other) const {
    if (cudaSuccess != cudaMemcpy(other.data(), data(),
                                  other.size() * sizeof(T),
                                  cudaMemcpyDeviceToHost)) {
      throw std::runtime_error("can't do device2host copy");
    }
  }

 private:
  size_type len_;
  std::unique_ptr<T[], cuda_deleter> ptr_;
};
