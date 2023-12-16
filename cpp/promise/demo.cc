#include <iostream>
#include <string>

#include "Promise.h"

int main(int argc, char *argv[]) {
  Promise<int> num;
  num.Then([](int num) { return num * num; })
      .Then([](int num) { return std::to_string(num); })
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

