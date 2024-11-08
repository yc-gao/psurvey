#pragma once

#include <utility>

template <class F>
class final_action {
 public:
  explicit final_action(const F& ff) noexcept : f{ff} {}
  explicit final_action(F&& ff) noexcept : f{std::move(ff)} {}

  ~final_action() noexcept {
    if (invoke) f();
  }

  final_action(final_action&& other) noexcept
      : f(std::move(other.f)), invoke(std::exchange(other.invoke, false)) {}

  final_action(const final_action&) = delete;
  void operator=(const final_action&) = delete;
  void operator=(final_action&&) = delete;

 private:
  F f;
  bool invoke = true;
};

// finally() - convenience function to generate a final_action
template <class F>
auto finally(F&& f) noexcept {
  return final_action<std::decay_t<F>>{std::forward<F>(f)};
}

#define _CAT(a, b) a##b
#define CAT(a, b) _CAT(a, b)
#define FINALLY(f) auto CAT(__FINALLY__, __COUNTER__) = finally(f)
