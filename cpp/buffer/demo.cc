#include <algorithm>
#include <cassert>
#include <cstring>

#include "Buffer.h"

template <typename T> void check() {
  T buf;
  {
    assert(buf.empty());
    Slice slice = buf.prepare(5);
    std::memcpy(slice.data(), "12345", slice.size());
    buf.commit(std::move(slice));
    assert(buf.size() == 5);
  }
  {
    Slice slice = buf.data();
    assert(slice.size() == 5);
    assert(!std::memcmp(slice.data(), "12345", slice.size()));
    buf.consume(std::move(slice));
    assert(buf.empty());
  }
  {
    assert(buf.empty());
    Slice slice = buf.prepare(5);
    std::memcpy(slice.data(), "12345", slice.size());
    buf.commit(std::move(slice));
    buf.clear();
    assert(buf.empty());
  }
}

int main(int argc, char *argv[]) {
  check<FixedBuffer<1024>>();
  check<DynamicBuffer<std::vector<char>>>();
  check<DynamicBuffer<std::string>>();
  check<DiscreteBuffer>();

  return 0;
}
