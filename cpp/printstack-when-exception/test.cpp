#include <stdexcept>
void func1() { throw std::runtime_error("func1"); }
void func0() { func1(); }
int main(int argc, char *argv[]) {
  func0();
  return 0;
}
