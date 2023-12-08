#include "stdio.h"

#include <chrono>
#include <thread>

void spin_sleep(std::chrono::nanoseconds dur) {
  auto now = std::chrono::steady_clock::now();
  auto end = now + dur;
  while (std::chrono::steady_clock::now() < end) {
  }
}

void demo11() {
  printf("demo11\n");
  spin_sleep(std::chrono::milliseconds(10));
}
void demo12() {
  printf("demo12\n");
  spin_sleep(std::chrono::milliseconds(5));
}

void demo() {
  for (int i = 0; i < 10; i++) {
    if (i & 1) {
      demo11();
    } else {
      demo12();
    }
  }
}
int main(int argc, char *argv[]) {
  for (int i = 0; i < 10; i++) {
    demo();
  }
  return 0;
}
