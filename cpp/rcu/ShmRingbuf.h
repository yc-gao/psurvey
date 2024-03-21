#pragma once

#include <cstdint>
#include <iterator>

struct ShmRingbuf {
  struct Segment {
    template <typename T = char> T *Addr() {
      return reinterpret_cast<T *>(data);
    }
    template <typename T = char> const T *Addr() const {
      return reinterpret_cast<const T *>(data);
    }
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
    using difference_type = ptrdiff_t;
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

  ShmRingbuf(std::uint64_t size) : head(0), capacity(size) {
    auto head = end();
    head->prev = 0;
    head->next = 0;
    head->size = 0;
  }

  SegmentIterator begin() { return std::next(SegmentIterator{data, head}); }
  SegmentIterator end() { return SegmentIterator{data, head}; }

  Segment *prepare(std::uint64_t size) {
    Segment *cur = reinterpret_cast<Segment *>(data + head);
    Segment *prev = reinterpret_cast<Segment *>(data + cur->prev);
    std::uint64_t prev_pos = reinterpret_cast<char *>(&*prev) - data;
    std::uint64_t prev_size = prev->size + sizeof(Segment);

    Segment *next = reinterpret_cast<Segment *>(data + cur->next);
    std::uint64_t next_pos = reinterpret_cast<char *>(&*prev) - data;
    std::uint64_t next_size = next->size + sizeof(Segment);

    auto capacity = this->capacity - sizeof(Segment);

    if (next_pos < prev_pos) {
      if (next_pos - prev_pos - prev_size >= size + sizeof(Segment)) {
        return reinterpret_cast<Segment *>(data + next_pos + next_size);
      }
      return nullptr;
    } else {
      if (capacity - next_pos - next_size >= size + sizeof(Segment)) {
        return reinterpret_cast<Segment *>(data + next_pos + next_size);
      }
      if (prev_pos >= size + sizeof(Segment)) {
        return reinterpret_cast<Segment *>(data);
      }
      return nullptr;
    }
  }
  void commit(std::uint64_t size) {
    Segment *segment = prepare(size);
    segment->size = size;

    Segment *head = reinterpret_cast<Segment *>(data + this->head);
    segment->prev = head->prev;
    segment->next = reinterpret_cast<char *>(&*head) - data;

    reinterpret_cast<Segment *>(data + segment->prev)->next =
        reinterpret_cast<char *>(&*segment) - data;
    reinterpret_cast<Segment *>(data + segment->next)->prev =
        reinterpret_cast<char *>(&*segment) - data;
  }

  void erase(SegmentIterator iter) {
    auto prev = std::prev(iter);
    auto next = std::next(iter);
    prev->next = reinterpret_cast<char *>(&*next) - data;
    next->prev = reinterpret_cast<char *>(&*prev) - data;
  }

  std::uint64_t head;
  std::uint64_t capacity;
  char data[];
};
