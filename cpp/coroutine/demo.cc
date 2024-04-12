#include <iostream>

#include "CoRunner.h"

int main(int argc, char *argv[]) {
  CoRunner runner;
  runner.Dispatch([&]() {
    std::cout << "coroutine0" << std::endl;
    runner.Dispatch([]() { std::cout << "coroutine1" << std::endl; });
  });
  runner.Dispatch([&]() {
    std::cout << "coroutine2" << std::endl;
    runner.Dispatch([]() { std::cout << "coroutine3" << std::endl; });
  });
  runner.Run();
  return 0;
}
