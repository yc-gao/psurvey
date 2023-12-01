#pragma once

#include "Buffer.h"

template <typename C, std::size_t alignment = 4096>
class DynamicBuffer : public Buffer {
  static_assert(sizeof(data_type) == sizeof(typename C::value_type));

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
    return {const_cast<data_type *>(
                reinterpret_cast<const data_type *>(buf_.data())),
            wpos_, size};
  }
  void commit(Slice slice) override { wpos_ += slice.size(); }

  Slice data(size_type size = 0) override {
    if (size == 0) {
      size = wpos_ - rpos_;
    }
    return {const_cast<data_type *>(
                reinterpret_cast<const data_type *>(buf_.data())),
            rpos_, std::min(size, wpos_ - rpos_)};
  }
  void consume(Slice slice) override {
    rpos_ += slice.size();
    if (rpos_ == wpos_) {
      rpos_ = wpos_ = 0;
    }
  }
};
