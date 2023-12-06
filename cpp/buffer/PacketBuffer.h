#pragma once

#include <cstdint>
#include <deque>

#include "Buffer.h"

class Span {
  void *data_;
  std::size_t size_;

public:
  Span(void *data, std::size_t size) : data_(data), size_(size) {}

  std::size_t size() const { return size_; }
  operator bool() const { return size(); }
  bool empty() const { return !size(); }

  void *data() { return data_; }
  const void *data() const { return data_; }

  auto begin() { return data(); }
  auto end() { return reinterpret_cast<std::uint8_t *>(data()) + size(); }
  auto begin() const { return data(); }
  auto end() const {
    return reinterpret_cast<const std::uint8_t *>(data()) + size();
  }
};

template <typename T> class PackedBuffer {
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

  const Span data() const { return {buf_.data(), packs_.front()}; }
  void consume(Span span) {
    buf_.consume(span.size());
    packs_.pop_front();
  }

  Span prepare(std::size_t size) { return {buf_.prepare(size), size}; }
  void commit(Span span) {
    buf_.commit(span.size());
    packs_.push_back(span.size());
  }

  std::size_t write(const void *data, std::size_t size) {
    Span span = prepare(size);
    std::memcpy(span.data(), data, size);
    commit(std::move(span));
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
