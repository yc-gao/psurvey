#include <cassert>
#include <string>
#include <vector>

#include "DynamicBuffer.h"
#include "FixedBuffer.h"

template <typename T> void check() {
  T buf;
  {
    assert(buf.empty());

    assert(4 == buf.write("1234", 4));
    assert(buf.size() == 4);
    buf.clear();

    buf << "1234";
    assert(buf.size() == sizeof("1234"));
    buf.clear();

    buf << "1234";
    assert(buf.size() == sizeof("1234"));
    assert(std::memcmp(buf.data(), "1234", sizeof("1234")) == 0);
    buf.consume(sizeof("1234"));
    assert(buf.empty());
  }
}

int main(int argc, char *argv[]) {
  check<FixedBuffer<1024>>();
  check<DynamicBuffer<std::vector<char>>>();
  check<DynamicBuffer<std::string>>();

  return 0;
}
