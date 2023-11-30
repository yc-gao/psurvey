#pragma once

#include "Buffer.h"

template <typename C, std::size_t alignment = 4096>
class DynamicBuffer : public Buffer {
  static_assert(sizeof(data_type) == sizeof(typename C::value_type));

  class ChunkImpl : public Chunk {
    C *buf_;
    size_type start_;
    size_type size_;

  public:
    ChunkImpl(C *buf, size_type start, size_type size)
        : buf_(buf), start_(start), size_(size) {}

    data_type *data() {
      return const_cast<data_type *>(
                 reinterpret_cast<const data_type *>(buf_->data())) +
             start_;
    }
    size_type size() const { return size_; }
  };

  C buf_;
  size_type wpos_{0};
  size_type rpos_{0};

public:
  size_type capacity() const override { return buf_.size() - wpos_; }
  size_type size() const override { return wpos_ - rpos_; }
  void clear() override { rpos_ = wpos_ = 0; }

  Slice prepare(size_type size) override {
    if (size + wpos_ > buf_.size()) {
      buf_.resize((size + wpos_ + alignment - 1) / alignment * alignment);
    }
    return {std::make_shared<ChunkImpl>(&buf_, wpos_, size)};
  }
  void commit(Slice slice) override { wpos_ += slice.size(); }

  Slice data() override {
    return {std::make_shared<ChunkImpl>(&buf_, rpos_, wpos_ - rpos_)};
  }
  void consume(Slice slice) override {
    rpos_ += slice.size();
    if (rpos_ == wpos_) {
      rpos_ = wpos_ = 0;
    }
  }
};
