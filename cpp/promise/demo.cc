#include <iostream>
#include <string>

#include "Promise.h"

struct Num {
  int val{0};
  Num(int val) : val(val) { std::cout << "Num(val)" << std::endl; }
  Num() { std::cout << "Num()" << std::endl; }
  Num(const Num &num) : val(num.val) { std::cout << "Num(&)" << std::endl; }
  Num(Num &&num) : val(num.val) { std::cout << "Num(&&)" << std::endl; }
};

struct Demo {
  int operator()() && { return 123; }
  void operator()() & {}
};

int main(int argc, char *argv[]) {
  {
    Promise<int> num;

    num.Then([](int num) { return Num(num); })
        .Then([](const Num &num) { return num.val * num.val; })
        .Then([](int num) { return std::to_string(num); })
        .Then([](const std::string &num) { std::cout << num << std::endl; });

    num.Resolve(100);
  }

  // {
  //   Promise<> a, b, c;
  //   a.Then([]() { std::cout << "a Resolved\n"; });
  //   b.Then([]() { std::cout << "b Resolved\n"; });
  //   c.Then([]() { std::cout << "c Resolved\n"; });
  //   (a + b + c).Then([]() { std::cout << "all resolved\n"; });
  //   a.Resolve();
  //   b.Resolve();
  //   c.Resolve();
  // }

  {
    Promise<> p;
    p.Then(Demo()).Then(
        [](int num) { std::cout << "num: " << num << std::endl; });
    Demo demo;
    p.Then(demo).Then([]() {});
    p.Resolve();
  }

  return 0;
}
