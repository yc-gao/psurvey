#pragma once

#include <cstddef>
#include <cstring>
#include <type_traits>

struct Buffer {
  virtual std::size_t size() const = 0;
  virtual std::size_t capacity() const = 0;
  virtual void clear() = 0;

  virtual const void *data() const = 0;
  virtual void consume(std::size_t) = 0;

  virtual void *prepare(std::size_t) = 0;
  virtual void commit(std::size_t) = 0;

  operator bool() const { return size(); }
  bool empty() const { return !size(); }

  std::size_t write(const void *data, std::size_t size) {
    std::memcpy(prepare(size), data, size);
    commit(size);
    return size;
  }
  template <typename T, typename = std::enable_if<std::is_pod<T>::value>>
  std::size_t write(const T &val) {
    return write(&val, sizeof(val));
  }
  template <typename T> friend Buffer &operator<<(Buffer &buf, const T &val) {
    buf.write(val);
    return buf;
  }
};
