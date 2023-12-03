#pragma once

#include <array>
#include <cstdint>

#include "Buffer.h"

template <std::size_t N> class FixedBuffer : public Buffer {
  std::array<std::uint8_t, N> buf_;
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

  void *prepare(std::size_t size) override { return buf_.data() + wpos_; }
  void commit(std::size_t size) override { wpos_ += size; }
};
