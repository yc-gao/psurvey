#include <cstdint>
#include <iostream>
#include <thread>

#include "PerfMonitor.h"

int fib(int n) {
  if (n == 1 || n == 0) {
    return 1;
  }
  return fib(n - 1) + fib(n - 2);
}
void do_perf0() {
  PerfMonitor monitor;

  std::uint64_t cycles;
  monitor.Monitor(PERF_COUNT_HW_CPU_CYCLES, 0, -1, &cycles);

  std::uint64_t insts;
  monitor.Monitor(PERF_COUNT_HW_INSTRUCTIONS, 0, -1, &insts);

  monitor.Begin();
  for (int i = 0; i < 10; i++) {
    fib(10);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    monitor.Update();
    std::cout << i << " cycles " << cycles << std::endl;
    std::cout << i << " insts " << insts << std::endl;
  }
  monitor.End();
}

int main(int argc, char *argv[]) {
  do_perf0();
  return 0;
}
