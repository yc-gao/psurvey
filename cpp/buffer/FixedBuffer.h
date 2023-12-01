#pragma once

#include <array>
#include <cstddef>
#include <memory>

#include "Buffer.h"

template <std::size_t N> class FixedBuffer : public Buffer {
  std::array<data_type, N> buf_;
  size_type wpos_{0};
  size_type rpos_{0};

public:
  size_type capacity() const override { return buf_.size() - wpos_; }
  size_type size() const override { return wpos_ - rpos_; }
  void clear() override { rpos_ = wpos_ = 0; }

  Slice prepare(size_type size) override {
    if (size + wpos_ > buf_.size()) {
      throw std::out_of_range("buf overflow");
    }
    return {buf_.data(), wpos_, size};
  }
  void commit(Slice slice) override { wpos_ += slice.size(); }

  Slice data(size_type size = 0) override {
    if (size == 0) {
      size = wpos_ - rpos_;
    }
    return {buf_.data(), rpos_, std::min(size, wpos_ - rpos_)};
  }
  void consume(Slice slice) override {
    rpos_ += slice.size();
    if (rpos_ == wpos_) {
      rpos_ = wpos_ = 0;
    }
  }
};
