#include <iostream>
#include <system_error>

enum class demo_errc {
  success = 0,
  invalid_arguments,
};

namespace std {

template <>
struct is_error_code_enum<demo_errc> : public true_type {};

inline error_code make_error_code(demo_errc __e) noexcept {
  class demo_category : public std::error_category {
   public:
    const char* name() const noexcept override { return "demo_ec"; }
    std::string message(int ev) const override { return "invalid arguments"; }
  };
  static demo_category tmp;
  return error_code(static_cast<int>(__e), tmp);
}

}  // namespace std

int main(int argc, char* argv[]) {
  auto ec = std::make_error_code(demo_errc::invalid_arguments);
  std::cout << ec << std::endl;
  std::cout << ec.message() << std::endl;
  return 0;
}
