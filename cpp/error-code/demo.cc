#include <system_error>
#include <type_traits>

enum class demo_errc {
  no_file = 10,
};

class demo_category : public std::error_category {
public:
  const char *name() const noexcept override { return "demo"; }
  std::string message(int e) const override {
    switch (static_cast<demo_errc>(e)) {
    case demo_errc::no_file:
      return "no such file";
    default:
      return "unkown";
    }
  }
};

namespace std {

template <> struct is_error_code_enum<demo_errc> : public std::true_type {};

inline std::error_code make_error_code(demo_errc e) {
  static demo_category category;
  return std::error_code(static_cast<int>(e), category);
}

}; // namespace std

#include <iostream>

int main(int argc, char *argv[]) {
  std::error_code ec = std::make_error_code(demo_errc::no_file);
  std::cerr << ec.message() << std::endl;
  return 0;
}
