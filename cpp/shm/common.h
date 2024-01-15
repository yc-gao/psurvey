#pragma once

#include <functional>

class Defer {
  std::function<void()> cb_;

public:
  Defer(const Defer &) = default;
  Defer(Defer &&) = default;
  Defer &operator=(const Defer &) = default;
  Defer &operator=(Defer &&) = default;

  ~Defer() {
    if (cb_) {
      cb_();
    }
  }
  template <typename F> Defer(F &&cb) : cb_(std::forward<F>(cb)) {}
};
#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)
#define MAKE_DEFER(func) Defer CONCATENATE(defer__, __LINE__)([&]() { func; })
