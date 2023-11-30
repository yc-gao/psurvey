#pragma once

#include "Buffer.h"
#include <queue>

template <typename T> class PacketBuffer : public Buffer {
  T buf_;

  std::queue<size_type> packs_;

public:
  size_type capacity() const override { return buf_.capacity(); }
  size_type size() const override { return packs_.size(); }
  void clear() override {
    using std::swap;
    std::queue<size_type> tmp;
    swap(tmp, packs_);
    buf_.clear();
  }

  Slice prepare(size_type size) override { return buf_.prepare(size); }
  void commit(Slice slice) override {
    packs_.push(slice.size());
    buf_.commit(std::move(slice));
  }

  Slice data() override {
    Slice tmp = buf_.data();
    return {tmp.chunk(), 0, packs_.front()};
  }
  void consume(Slice slice) override {
    packs_.pop();
    buf_.consume(std::move(slice));
  }
};
