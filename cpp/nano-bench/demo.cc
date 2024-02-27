#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

int test_func() {
  int s = 0;
  for (int i = 0; i < 100; i++) {
    s += i;
  };
  return s;
}

void do_nanobench() {
  ankerl::nanobench::Bench().run("some double ops", [] { test_func(); });
}

int main() {
  do_nanobench();
  return 0;
}
