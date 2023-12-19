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

int main(int argc, char *argv[]) {
  Promise<int> num;

  auto square =
      num.Then([](int num) { return Num(num); }).Then([](const Num &num) {
        return num.val * num.val;
      });

  square.Then([](const Num &num) { return std::to_string(num.val); })
      .Then([](std::string num) { std::cout << "num: " << num << std::endl; })
      .Catch([](const std::error_code &ec) {
        std::cerr << "error msg: " << ec.message() << std::endl;
      })
      .Finally([]() { std::cout << "finally1\n"; })
      .Finally([]() { std::cout << "finally2\n"; });

  num.Resolve(100);
  num.Reject(std::error_code());

  Promise<> a, b, c;
  a.Then([]() { std::cout << "a Resolved\n"; });
  b.Then([]() { std::cout << "b Resolved\n"; });
  c.Then([]() { std::cout << "c Resolved\n"; });
  Promise<> r = a + b + c;
  r.Then([]() { std::cout << "c\n"; });
  a.Resolve();
  b.Resolve();
  c.Resolve();
  return 0;
}
