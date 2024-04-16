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
  ringbuf->commit(seg, sizeof(val));
  return true;
}

bool running{true};
char buf[1024];

int main(int argc, char *argv[]) {
  std::signal(SIGINT, [](int) { running = false; });
  ShmRingbuf *ringbuf = new (buf) ShmRingbuf(sizeof(buf) - sizeof(ShmRingbuf));

  for (std::uint64_t i = 0; running; i++) {
    if (!ringbuf_append(ringbuf, i)) {
      std::cout << "append failed, erase\n";
      ringbuf->erase(ringbuf->begin());
      ringbuf_append(ringbuf, i);
    }
    for (auto beg = std::make_reverse_iterator(ringbuf->end()),
              end = std::make_reverse_iterator(ringbuf->begin());
         beg != end; beg++) {
      std::cout << beg->Value<std::uint64_t>() << '\n';
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  return 0;
}
