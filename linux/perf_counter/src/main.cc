#include <csignal>
#include <cstdint>
#include <iostream>
#include <thread>

#include "PerfMonitor.h"
#include "Rate.h"

bool running{true};

void do_perf0(int pid) {
  PerfMonitor monitor;

  std::uint64_t insts;
  monitor.Monitor(PERF_COUNT_HW_INSTRUCTIONS, pid, -1, &insts);

  monitor.Begin();
  Rate rate(100);
  while (running && rate.Ok()) {
    monitor.Update();
    std::cout << insts << '\n';
  }
  monitor.End();
}

int main(int argc, char *argv[]) {
  std::signal(SIGINT, [](int) { running = false; });

  int pid = 0;
  if (argc == 2) {
    pid = atoi(argv[1]);
  }
  do_perf0(pid);
  return 0;
}
