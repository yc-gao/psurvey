int fib(int a) {
  if (a < 2) {
    return 1;
  }
  return fib(a - 1) + fib(a - 2);
}

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("fib") {
  CHECK(fib(0) == 1);
  CHECK(fib(1) == 1);
  CHECK(fib(2) == 2);
}
