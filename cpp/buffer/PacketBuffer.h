#pragma once

#include <cstdint>
#include <deque>

#include "Buffer.h"

class Slice {
  void *data_;
  std::size_t size_;

public:
  Slice(void *data, std::size_t size) : data_(data), size_(size) {}

  operator bool() const { return size(); }
  bool empty() const { return !size(); }

  std::size_t size() const { return size_; }
  void *data() { return data_; }
  const void *data() const { return data_; }

  Slice sub(std::size_t offset) {
    return {reinterpret_cast<std::uint8_t *>(data_) + offset, size_ - offset};
  }

  Slice sub(std::size_t offset, std::size_t size) {
    return {reinterpret_cast<std::uint8_t *>(data_) + offset, size};
  }
};

template <typename T> class PacketBuffer {
  T buf_;
  std::deque<std::size_t> packs_;

public:
  operator bool() const { return size(); }
  bool empty() const { return !size(); }

  std::size_t size() const { return packs_.size(); }
  void clear() {
    buf_.clear();
    packs_.clear();
  }

  const Slice data() const { return {buf_.data(), packs_.front()}; }
  void consume(Slice slice) {
    buf_.consume(slice.size());
    packs_.pop_front();
  }

  Slice prepare(std::size_t size) { return {buf_.prepare(size), size}; }
  void commit(Slice slice) {
    buf_.commit(slice.size());
    packs_.push_back(slice.size());
  }

  std::size_t write(const void *data, std::size_t size) {
    Slice slice = prepare(size);
    std::memcpy(slice.data(), data, size);
    commit(std::move(slice));
    return size;
  }
  template <typename U, typename = std::enable_if<std::is_pod<U>::value>>
  std::size_t write(const U &val) {
    return write(&val, sizeof(val));
  }
  template <typename U> friend Buffer &operator<<(Buffer &buf, const U &val) {
    buf.write(val);
    return buf;
  }
};
