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
  num.Then([](int num) { return Num(num); })
      .Then([](Num num) { return num.val * num.val; })
      .Then([](Num num) { return std::to_string(num.val); })
      .Then([](std::string num) { std::cout << "num: " << num << std::endl; })
      .Catch([](const std::error_code &ec) {
        std::cerr << "error msg: " << ec.message() << std::endl;
      })
      .Finally([]() { std::cout << "finally1\n"; })
      .Finally([]() { std::cout << "finally2\n"; });
  ;

  num.Resolve(100);
  num.Reject(std::error_code());
  return 0;
}
