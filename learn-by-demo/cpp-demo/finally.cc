#pragma once

#include <algorithm>
#include <utility>

#if defined(__cplusplus) && (__cplusplus >= 201703L)
#define GSL_NODISCARD [[nodiscard]]
#else
#define GSL_NODISCARD
#endif  // defined(__cplusplus) && (__cplusplus >= 201703L)

template <class F>
class final_action {
 public:
  ~final_action() noexcept {
    if (invoke) f();
  }

  final_action(const final_action&) = delete;
  final_action(final_action&&) = delete;
  void operator=(const final_action&) = delete;
  void operator=(final_action&&) = delete;

  explicit final_action(const F& ff) noexcept : f{ff} {}
  explicit final_action(F&& ff) noexcept : f{std::move(ff)} {}

 private:
  F f;
  bool invoke = true;
};

// finally() - convenience function to generate a final_action
template <class F>
GSL_NODISCARD auto finally(F&& f) noexcept {
  return final_action<std::decay_t<F>>{std::forward<F>(f)};
}

#include <doctest/doctest.h>

TEST_CASE("testing HandleHolder") {
  int num = 0;
  {
    finally([&]() { num++; });
  }
  CHECK(num == 1);
}
