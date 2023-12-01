#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

#include "BoundedPacketBuffer.h"
#include "DynamicBuffer.h"
#include "FixedBuffer.h"
#include "PacketBuffer.h"

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
  {
    buf << "1234";
    assert(buf.size() == sizeof("1234"));
    std::vector<int> vs{1, 2, 3};
    buf << vs;
    assert(buf.size() == sizeof("1234") + sizeof(int) * 3);
  }
}

template <typename T> void check_packet() {
  T buf;
  {
    assert(buf.empty());
    Slice slice = buf.prepare(5);
    std::memcpy(slice.data(), "12345", slice.size());
    buf.commit(std::move(slice));
    assert(buf.size() == 1);
  }
  {
    Slice slice = buf.packet();
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

template <typename T> void check_bounded() {
  check_packet<T>();
  T buf(3);
  {
    assert(buf.empty());
    Slice slice = buf.prepare(5);
    std::memcpy(slice.data(), "12345", slice.size());
    buf.commit(std::move(slice));
    assert(buf.size() == 1);
  }
  {
    Slice slice = buf.prepare(5);
    std::memcpy(slice.data(), "12345", slice.size());
    buf.commit(std::move(slice));
    assert(buf.size() == 2);
  }
  {
    Slice slice = buf.prepare(5);
    std::memcpy(slice.data(), "12345", slice.size());
    buf.commit(std::move(slice));
    assert(buf.size() == 3);
  }
  {
    Slice slice = buf.prepare(5);
    std::memcpy(slice.data(), "12345", slice.size());
    buf.commit(std::move(slice));
    assert(buf.size() == 3);
  }
}

int main(int argc, char *argv[]) {
  check<FixedBuffer<1024>>();
  check<DynamicBuffer<std::vector<char>>>();
  check<DynamicBuffer<std::string>>();

  check_packet<PacketBuffer<DynamicBuffer<std::string>>>();
  check_packet<PacketBuffer<DynamicBuffer<std::vector<char>>>>();

  check_bounded<BoundedPacketBuffer<DynamicBuffer<std::string>>>();
  check_bounded<BoundedPacketBuffer<DynamicBuffer<std::vector<char>>>>();

  return 0;
}
