#pragma once

#include <chrono>
#include <cstdint>
#include <thread>

class Rate {
  std::chrono::nanoseconds period;

  std::chrono::time_point<std::chrono::steady_clock> tm;

public:
  Rate(std::uint64_t freq) : period(1000000000 / freq), tm() {}

  bool Ok() {
    auto now = std::chrono::steady_clock::now();
    if ((now - tm) >= period) {
      tm = now;
      return true;
    } else {
      std::this_thread::sleep_for(period - (now - tm));
      return Ok();
    }
  }
};
