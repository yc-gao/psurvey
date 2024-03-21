#include <csignal>
#include <iostream>
#include <thread>
#include <type_traits>

#include "ShmRingbuf.h"

template <typename T> bool ringbuf_append(ShmRingbuf *ringbuf, const T &val) {
  auto seg = ringbuf->prepare(sizeof(val));
  if (!seg) {
    return false;
  }
  seg->Value<std::decay_t<T>>() = val;
  ringbuf->commit(sizeof(val));
  return true;
}

bool running{true};
char buf[1000];
int main(int argc, char *argv[]) {
  std::signal(SIGINT, [](int) { running = false; });
  ShmRingbuf *ringbuf = new (buf) ShmRingbuf(sizeof(buf) - sizeof(ShmRingbuf));

  for (std::uint64_t i = 1; i < 4; i++) {
    ringbuf_append(ringbuf, i);
  }

  for (auto beg = ringbuf->begin(), end = ringbuf->end(); beg != end; ++beg) {
    std::cout << beg->Value<int>() << std::endl;
  }

  ringbuf->erase(ringbuf->begin());

  for (auto beg = ringbuf->begin(), end = ringbuf->end(); beg != end; ++beg) {
    std::cout << beg->Value<int>() << std::endl;
  }

  // std::thread t1([ringbuf]() {
  //   int n = 1;
  //   while (running) {
  //     if (!ringbuf_append(ringbuf, n)) {
  //       ringbuf->rcu_remove_if([n](ShmRingbuf::Segment &seg) {
  //         return !seg.size || seg.Value<int>() < n - 5;
  //       });
  //       continue;
  //     }
  //     // std::cout << "append " << n << '\n';
  //     n++;
  //   }
  // });
  // std::thread t2([ringbuf]() {
  //   while (running) {
  //     ringbuf->rcu_lock();
  //     for (auto beg = std::make_reverse_iterator(ringbuf->end()),
  //               end = std::make_reverse_iterator(ringbuf->begin());
  //          beg != end; ++beg) {
  //       if (beg->size) {
  //         std::cout << beg->Value<int>() << '\n';
  //         break;
  //       }
  //     }
  //     ringbuf->rcu_unlock();
  //   }
  // });
  //
  // t1.join();
  // t2.join();
  return 0;
}
