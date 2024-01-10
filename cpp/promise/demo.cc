#include <iostream>
#include <string>
#include <system_error>

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
        .Then([](const Num &num) { return num.val + 1; })
        .Then([](int num) { return std::to_string(num); })
        .Then([](const std::string &num) { std::cout << num << std::endl; })
        .Finally([]() { std::cout << "Finally0" << std::endl; });
    num.Resolve(100);
  }
  {
    Promise<int> num;
    num.Then([](int num) { return Num(num); })
        .Then([](const Num &num) { return num.val * num.val; })
        .Then([](int num) { return std::to_string(num); })
        .Then([](const std::string &num) { std::cout << num << std::endl; })
        .Catch([](const std::error_code &ec) {
          std::cout << "ec: " << ec.message() << std::endl;
        })
        .Finally([]() { std::cout << "Finally1" << std::endl; });
    num.Reject(std::make_error_code(std::errc::invalid_argument));
  }
  {
    Promise<> p;
    p.Then(Demo()).Then(
        [](int num) { std::cout << "num: " << num << std::endl; });
    Demo demo;
    p.Then(demo).Then([]() { std::cout << "empty" << std::endl; });
    p.Resolve();
  }
  {
    Promise<> a, b, c;
    a.Then([]() { std::cout << "a" << std::endl; });
    b.Then([]() { std::cout << "b" << std::endl; });
    c.Then([]() { std::cout << "c" << std::endl; });
    a.And(b).And(c).Then([]() { std::cout << "all" << std::endl; });
    a.Resolve();
    b.Resolve();
    c.Resolve();
  }

  return 0;
}
