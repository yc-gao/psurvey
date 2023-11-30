#pragma once

#include <array>
#include <cstddef>
#include <memory>

#include "Buffer.h"

template <std::size_t N> class FixedBuffer : public Buffer {
  class ChunkImpl : public Chunk {
    data_type *data_;
    size_type size_;

  public:
    ChunkImpl(data_type *data, size_type size) : data_(data), size_(size) {}

    data_type *data() { return data_; }
    size_type size() const { return size_; }
  };

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
    return {std::make_shared<ChunkImpl>(buf_.data() + wpos_, size)};
  }
  void commit(Slice slice) override { wpos_ += slice.chunk()->size(); }

  Slice data() override {
    return {std::make_shared<ChunkImpl>(buf_.data() + rpos_, wpos_ - rpos_)};
  }
  void consume(Slice slice) override {
    rpos_ += slice.chunk()->size();
    if (rpos_ == wpos_) {
      rpos_ = wpos_ = 0;
    }
  }
};
