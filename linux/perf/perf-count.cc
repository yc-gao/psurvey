#include "common.h"

int main(int argc, char *argv[]) {
  std::signal(SIGINT, [](int) { running = false; });
  std::signal(SIGTERM, [](int) { running = false; });
  // return do_perf0(argc, argv);
  // return do_perf1(argc, argv);
  return do_perf2(argc, argv);
  return 0;
}
