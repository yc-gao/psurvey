#pragma once

#include <cstdint>

#include "Buffer.h"

template <typename T> class DynamicBuffer : public Buffer {
  static_assert(sizeof(std::uint8_t) == sizeof(typename T::value_type));
  static constexpr std::size_t alignment = 4096;

  T buf_;
  std::size_t rpos_{0};
  std::size_t wpos_{0};

public:
  std::size_t size() const override { return wpos_ - rpos_; }
  std::size_t capacity() const override { return buf_.size() - wpos_; }
  void clear() override { rpos_ = wpos_ = 0; }

  const void *data() const override { return buf_.data() + rpos_; }
  void consume(std::size_t size) override {
    rpos_ += size;
    if (rpos_ == wpos_) {
      rpos_ = wpos_ = 0;
    }
  }

  void *prepare(std::size_t size) override {
    if (wpos_ + size > buf_.size()) {
      buf_.resize((wpos_ + size + alignment - 1) / alignment * alignment);
    }
    return buf_.data() + wpos_;
  }
  void commit(std::size_t size) override { wpos_ += size; }
};
