#pragma once

#include "PacketBuffer.h"

template <typename C> class BoundedPacketBuffer : public Buffer {

  PacketBuffer<C> buf_;

  size_type capacity_;

public:
  BoundedPacketBuffer(size_type capacity = 1) : capacity_(capacity) {}

  size_type capacity() const override { return capacity_ - buf_.size(); }
  size_type size() const override { return buf_.size(); }
  void clear() override { return buf_.clear(); }

  Slice prepare(size_type size) override { return buf_.prepare(size); }
  void commit(Slice slice) override {
    if (size() < capacity_) {
      buf_.commit(std::move(slice));
    }
  }

  Slice data(size_type size = 0) override {
    throw std::logic_error("unreachable code");
    return {};
  }
  Slice packet() { return buf_.packet(); }
  void consume(Slice slice) override { return buf_.consume(std::move(slice)); }
};
