#include "stacktrace.h"

void func2() { print_stack(); }
void func1() { func2(); }

void func0() { func1(); }
int main(int argc, char *argv[]) {
  func0();
  return 0;
}
