#pragma once

#include <cstring>
#include <functional>
#include <memory>
#include <vector>

class Chunk {
protected:
  using size_type = std::size_t;
  using data_type = std::uint8_t;

public:
  const data_type *data() const { return const_cast<Chunk *>(this)->data(); }

  virtual data_type *data() = 0;
  virtual size_type size() const = 0;
};

class Slice {
  using size_type = std::size_t;
  using data_type = std::uint8_t;

  using iterator = data_type *;
  using const_iterator = const data_type *;

  std::shared_ptr<Chunk> chunk_;
  data_type *data_{nullptr};
  size_type size_{0};

public:
  Slice() = default;
  Slice(const Slice &other)
      : chunk_(other.chunk_), data_(other.data_), size_(other.size_) {}
  Slice(Slice &&other) {
    Slice tmp;
    swap(tmp, other);
    swap(*this, tmp);
  }
  Slice &operator=(Slice other) {
    swap(*this, other);
    return *this;
  }

  Slice(std::shared_ptr<Chunk> chunk)
      : Slice(chunk, (size_type)0, chunk->size()) {}
  Slice(std::shared_ptr<Chunk> chunk, size_type start, size_type size)
      : Slice(chunk, chunk->data() + start, size) {}

  Slice(data_type *data, size_type start, size_type size)
      : Slice(data + start, size) {}
  Slice(data_type *data, size_type size) : Slice(nullptr, data, size) {}

  Slice(std::shared_ptr<Chunk> chunk, data_type *data, size_type size)
      : chunk_(std::move(chunk)), data_(data), size_(size) {}

  friend void swap(Slice &lhs, Slice &rhs) {
    using std::swap;
    swap(lhs.chunk_, rhs.chunk_);
    swap(lhs.data_, rhs.data_);
    swap(lhs.size_, rhs.size_);
  }

  void clear() {
    Slice tmp;
    swap(*this, tmp);
  }

  std::shared_ptr<Chunk> chunk() { return chunk_; }

  const data_type *data() const { return const_cast<Slice *>(this)->data(); }
  data_type *data() { return data_; }
  size_type size() const { return size_; }
  bool empty() const { return !size(); }
  operator bool() const { return !empty(); }

  iterator begin() { return data_; }
  const_iterator begin() const { return data_; }
  iterator end() { return data_ + size(); }
  const_iterator end() const { return data_ + size(); }
};

class Buffer {
protected:
  using size_type = std::size_t;
  using data_type = std::uint8_t;

public:
  bool empty() const { return !size(); }

  virtual size_type capacity() const = 0;
  virtual size_type size() const = 0;
  virtual void clear() = 0;

  virtual std::size_t write(const void *data, std::size_t size) {
    Slice slice = prepare(size);
    std::memcpy(slice.data(), data, slice.size());
    commit(std::move(slice));
    return size;
  }

  virtual Slice prepare(size_type) = 0;
  virtual void commit(Slice) = 0;

  virtual Slice data(size_type = 0) = 0;
  virtual void consume(Slice) = 0;

  template <typename T, typename = std::enable_if<std::is_pod<T>::value>>
  friend Buffer &operator<<(Buffer &buffer, const T &val) {
    buffer.write(&val, sizeof(val));
    return buffer;
  }
  template <typename CharT, typename Traits, typename Allocator>
  friend Buffer &
  operator<<(Buffer &buffer,
             const std::basic_string<CharT, Traits, Allocator> &val) {
    buffer.write(val.data(), val.size());
    return buffer;
  }
  template <typename T, typename Allocator>
  friend Buffer &operator<<(Buffer &buffer,
                            const std::vector<T, Allocator> &vals) {
    for (auto &&v : vals) {
      buffer << v;
    }
    return buffer;
  }
};
