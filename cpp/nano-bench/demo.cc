#include <cstdlib>
#include <unistd.h>

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

int main() {
  ankerl::nanobench::Bench().run("malloc", [] { malloc(8); });
  ankerl::nanobench::Bench().run("brk", [] { sbrk(8); });
  return 0;
}
