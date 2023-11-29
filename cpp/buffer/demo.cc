#include <algorithm>
#include <cassert>

#include "Buffer.h"

int main(int argc, char *argv[]) {
  FixedBuffer<1024> fixed_buf;
  {
    Slice slice = fixed_buf.prepare(512);
    std::fill(slice.begin(), slice.end(), 1);
    fixed_buf.commit(slice);
  }

  {
    Slice slice = fixed_buf.data();
    assert(fixed_buf.capacity() == 512);
    assert(std::all_of(slice.begin(), slice.end(),
                       [](const auto &item) { return item == 1; }));
    fixed_buf.consume(std::move(slice));
    assert(fixed_buf.capacity() == 1024);
  }

  DiscreteBuffer dis_buf;
  {
    Slice slice = dis_buf.prepare(512);
    std::fill(slice.begin(), slice.end(), 1);
    dis_buf.commit(slice);
  }
  {
    Slice slice = dis_buf.data();
    assert(std::all_of(slice.begin(), slice.end(),
                       [](const auto &item) { return item == 1; }));
    dis_buf.consume(std::move(slice));
  }
  return 0;
}
