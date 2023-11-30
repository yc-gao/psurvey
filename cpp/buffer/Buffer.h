#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

class Chunk {
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
  Slice(const Slice &) = default;
  Slice(Slice &&) = default;
  Slice &operator=(const Slice &) = default;
  Slice &operator=(Slice &&) = default;

  Slice(std::shared_ptr<Chunk> chunk) : Slice(chunk, 0, chunk->size()) {}
  Slice(std::shared_ptr<Chunk> chunk, size_type start, size_type size)
      : chunk_(std::move(chunk)), data_(chunk_->data() + start), size_(size) {
    if (start + size > chunk_->size()) {
      throw std::invalid_argument("chunk overflow");
    }
  }

  std::shared_ptr<Chunk> chunk() { return chunk_; }
  operator bool() const { return !empty(); }

  const data_type *data() const { return const_cast<Slice *>(this)->data(); }
  data_type *data() { return data_; }
  size_type size() const { return size_; }
  bool empty() const { return !size(); }

  iterator begin() { return data_; }
  const_iterator begin() const { return data_; }
  iterator end() { return data_ + size(); }
  const_iterator end() const { return data_ + size(); }
};

template <std::size_t N> class FixedBuffer {
  using size_type = std::size_t;
  using data_type = std::uint8_t;

  class ChunkImpl : public Chunk {
    using size_type = std::size_t;
    using data_type = std::uint8_t;

    data_type *data_;
    size_type size_;

  public:
    ChunkImpl(data_type *data, size_type size) : data_(data), size_(size) {}
    data_type *data() override { return data_; }
    size_type size() const override { return size_; }
  };

  std::array<data_type, N> buf_;
  size_type rpos_{0};
  size_type wpos_{0};

public:
  size_type capacity() const { return buf_.size() - wpos_; }
  size_type size() const { return wpos_ - rpos_; }
  bool empty() const { return !size(); }

  Slice prepare(size_type size) {
    if (size > buf_.size() - wpos_) {
      throw std::out_of_range("buf overflow");
    }
    return {std::make_shared<ChunkImpl>(buf_.data() + wpos_, size)};
  }
  void commit(Slice slice) { wpos_ += slice.chunk()->size(); }

  Slice data() {
    return {std::make_shared<ChunkImpl>(buf_.data() + rpos_, wpos_ - rpos_)};
  }
  void consume(Slice slice) {
    rpos_ += slice.chunk()->size();
    if (rpos_ == wpos_) {
      rpos_ = wpos_ = 0;
    }
  }
};

template <typename C> class DynamicBuffer {
  using size_type = std::size_t;
  using data_type = std::uint8_t;

  static constexpr size_type alignment = 4096;
  // TODO:
  static_assert(sizeof(data_type) == sizeof(typename C::value_type), "");

  class ChunkImpl : public Chunk {
    using size_type = std::size_t;
    using data_type = std::uint8_t;

    C *buf_;
    size_type start_;
    size_type size_;

  public:
    ChunkImpl(C *buf, size_type start, size_type size)
        : buf_(buf), start_(start), size_(size) {}

    data_type *data() override {
      return const_cast<data_type *>(
                 reinterpret_cast<const data_type *>(buf_->data())) +
             start_;
    }
    size_type size() const override { return size_; }
  };

  C buf_;
  size_type rpos_{0};
  size_type wpos_{0};

public:
  size_type capacity() const { return buf_.size() - wpos_; }
  size_type size() const { return wpos_ - rpos_; }
  bool empty() const { return !size(); }

  Slice prepare(size_type size) {
    if (size > buf_.size() - wpos_) {
      size_type capacity = size + wpos_;
      capacity = (capacity + alignment - 1) / alignment * alignment;
      buf_.resize(capacity);
    }
    return {std::make_shared<ChunkImpl>(&buf_, wpos_, size)};
  }
  void commit(Slice slice) { wpos_ += slice.chunk()->size(); }

  Slice data() {
    return {std::make_shared<ChunkImpl>(&buf_, rpos_, wpos_ - rpos_)};
  }
  void consume(Slice slice) {
    rpos_ += slice.chunk()->size();
    if (rpos_ == wpos_) {
      rpos_ = wpos_ = 0;
    }
  }
};

class DiscreteBuffer {
  using size_type = std::size_t;
  using data_type = std::uint8_t;
  static constexpr size_type alignment = 4096;

  class ChunkImpl : public Chunk {
    using size_type = std::size_t;
    using data_type = std::uint8_t;

    std::vector<data_type> buf_;
    size_type capacity_{0};
    size_type size_{0};

    std::function<void()> destruct_cb_;

  public:
    virtual ~ChunkImpl() {
      if (destruct_cb_) {
        destruct_cb_();
      }
    }
    void OnDestruct() {
      using std::swap;
      std::function<void()> tmp;
      swap(tmp, destruct_cb_);
    }
    template <typename F> void OnDestruct(F &&func) {
      destruct_cb_ = std::forward<F>(func);
    }
    friend void swap(ChunkImpl &lhs, ChunkImpl &rhs) {
      using std::swap;
      swap(lhs.buf_, rhs.buf_);
      swap(lhs.capacity_, rhs.capacity_);
      swap(lhs.size_, rhs.size_);
      swap(lhs.destruct_cb_, rhs.destruct_cb_);
    }
    size_type capacity() const { return capacity_; }
    void resize(size_type size) {
      size_ = size;
      if (size > capacity()) {
        size = (size + alignment - 1) / alignment * alignment;
        buf_.resize(size);
      }
    }

    data_type *data() override { return buf_.data(); }
    size_type size() const override { return size_; }
  };
  struct ChunkCompare {
    bool operator()(const std::shared_ptr<ChunkImpl> &lhs,
                    const std::shared_ptr<ChunkImpl> &rhs) const {
      return lhs->capacity() < rhs->capacity();
    }
  };

  std::set<std::shared_ptr<ChunkImpl>, ChunkCompare> reserved_;
  std::vector<std::shared_ptr<ChunkImpl>> commited_;

public:
  size_type size() const {
    return std::accumulate(
        commited_.begin(), commited_.end(), 0,
        [](size_type size, const auto &item) { return item->size() + size; });
  }
  bool empty() const { return !size(); }

  Slice prepare(size_type size) {
    std::shared_ptr<ChunkImpl> chunk;
    auto iter = std::lower_bound(
        reserved_.begin(), reserved_.end(), size,
        [](const std::shared_ptr<ChunkImpl> &chunk, size_type size) {
          return chunk->capacity() < size;
        });
    if (iter == reserved_.end()) {
      chunk = std::make_shared<ChunkImpl>();
      chunk->resize(size);
    } else {
      chunk = *iter;
      reserved_.erase(iter);
      chunk->resize(size);
    }
    chunk->OnDestruct([this, chunk = chunk.get()]() {
      auto tmp = std::make_shared<ChunkImpl>();
      swap(*chunk, *tmp);
      tmp->OnDestruct();
      reserved_.emplace(std::move(tmp));
    });
    return {chunk};
  }
  void commit(Slice slice) {
    auto chunk = std::dynamic_pointer_cast<ChunkImpl>(slice.chunk());
    chunk->OnDestruct();
    commited_.emplace_back(std::move(chunk));
  }
  Slice data() {
    if (!commited_.empty()) {
      return {commited_.front()};
    } else {
      return {};
    }
  }
  void consume(Slice slice) {
    if (slice.chunk() != commited_.front()) {
      throw std::logic_error("unknown chunk");
    }
    reserved_.emplace(commited_.front());
    commited_.erase(commited_.begin());
  }
};
