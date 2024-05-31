#include <iostream>
int fib(int);
int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  std::cout << "fib(10): " << fib(10) << std::endl;
  return 0;
}
