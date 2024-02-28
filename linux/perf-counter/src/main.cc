#include <csignal>
#include <cstdint>
#include <iostream>

#include "PerfMonitor.h"
#include "Rate.h"

bool running{true};

void do_perf0(int pid) {
  PerfMonitor monitor;

  std::uint64_t inst;
  monitor.Monitor(PERF_COUNT_HW_INSTRUCTIONS, pid, -1, &inst);

  std::uint64_t nanosleep;
  monitor.Monitor(PERF_TYPE_TRACEPOINT, 379, pid, -1, &nanosleep); // nanosleep

  monitor.Begin();
  Rate rate(1000);
  while (running && rate.Ok()) {
    monitor.Update();
    std::cout << inst << '\t' << nanosleep << '\n';
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
