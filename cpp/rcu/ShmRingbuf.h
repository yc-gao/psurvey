#pragma once

#include <atomic>
#include <cstdint>
#include <iterator>

struct ShmRingbuf {
  struct Segment {
    template <typename T> T &Value() { return *reinterpret_cast<T *>(data); }
    template <typename T> const T &Value() const {
      return *reinterpret_cast<const T *>(data);
    }

    std::uint64_t prev;
    std::uint64_t next;
    std::uint64_t size;
    char data[];
  };

  struct SegmentIterator {
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = int;
    using value_type = Segment;
    using pointer = value_type *;
    using reference = value_type &;

    SegmentIterator(char *base, std::uint64_t pos) : base(base), pos(pos) {}

    Segment &operator*() { return *reinterpret_cast<Segment *>(base + pos); }
    Segment *operator->() { return reinterpret_cast<Segment *>(base + pos); }

    bool operator==(SegmentIterator other) const {
      return base == other.base && pos == other.pos;
    }
    bool operator!=(SegmentIterator other) const { return !(*this == other); }

    SegmentIterator &operator++() {
      pos = (**this).next;
      return *this;
    }
    SegmentIterator operator++(int) {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }
    SegmentIterator &operator--() {
      pos = (**this).prev;
      return *this;
    }
    SegmentIterator operator--(int) {
      auto tmp = *this;
      --(*this);
      return tmp;
    }

    char *base;
    std::uint64_t pos;
  };

  ShmRingbuf(std::uint64_t size)
      : tail(0), head(sizeof(Segment)), capacity(size) {
    auto tail = begin();
    tail->next = this->head;
    tail->size = 0;

    auto head = end();
    head->prev = this->tail;
    head->size = 0;
  }

  void rcu_lock() { rcu_count.fetch_add(1); }
  void rcu_unlock() { rcu_count.fetch_sub(1); }
  void rcu_sync() {
    while (rcu_count) {
      // WARNING: maybe sleep
    }
  }

  SegmentIterator begin() { return SegmentIterator{data, tail}; }
  SegmentIterator end() { return SegmentIterator{data, head}; }

  template <typename F> void remove_if(F &&op) {
    auto st = begin(), ed = end();
    for (; st != ed; ++st) {
      if (!op(*st)) {
        break;
      }
    }
    // WARNING: keep 2 elems
    if (st == ed) {
      --st;
    }
    tail = reinterpret_cast<char *>(&*st) - data;
  }

  template <typename F> void rcu_remove_if(F &&f) {
    remove_if(std::forward<F>(f));
    rcu_sync();
  }

  Segment *prepare(std::uint64_t size) {
    if (head < tail) {
      if (tail - head >= size + sizeof(Segment) * 2) {
        return reinterpret_cast<Segment *>(data + head);
      }
      return nullptr;
    } else {
      if (capacity - head >= size + sizeof(Segment) * 2) {
        return reinterpret_cast<Segment *>(data + head);
      }
      if (tail >= size + sizeof(Segment) * 2) {
        return reinterpret_cast<Segment *>(data);
      }
      return nullptr;
    }
  }
  void commit(std::uint64_t size) {
    Segment *cur_seg = prepare(size);
    cur_seg->size = size;

    Segment *head_seg = reinterpret_cast<Segment *>(data + head);
    Segment *next_seg = (Segment *)((char *)cur_seg + sizeof(Segment) + size);
    next_seg->size = 0;

    next_seg->prev = (char *)cur_seg - data;
    cur_seg->next = (char *)next_seg - data;
    if (cur_seg != head_seg) {
      head_seg->next = (char *)cur_seg - data;
      cur_seg->prev = (char *)head_seg - data;
    }
    head = cur_seg->next;
  }

  std::atomic_int rcu_count{0};
  std::uint64_t tail;
  std::uint64_t head;
  std::uint64_t capacity;
  char data[];
};
