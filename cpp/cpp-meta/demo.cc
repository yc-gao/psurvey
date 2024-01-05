#include <type_traits>
#include <utility>

template <typename T,
          typename U = typename std::remove_cv_t<std::remove_reference_t<T>>>
T make_var() {
  static U tmp;
  return tmp;
}

void var_test() {
  {
    int n;
    static_assert(std::is_same_v<decltype(n), int>);
    static_assert(std::is_same_v<decltype((n)), int &>);
  }
  {
    const int n = 1;
    static_assert(std::is_same_v<decltype(n), const int>);
    static_assert(std::is_same_v<decltype((n)), const int &>);
  }

  {
    int a;
    int &n = a;
    static_assert(std::is_same_v<decltype(n), int &>);
    static_assert(std::is_same_v<decltype((n)), int &>);
  }
  {
    int a;
    const int &n = a;
    static_assert(std::is_same_v<decltype(n), const int &>);
    static_assert(std::is_same_v<decltype((n)), const int &>);
  }

  {
    int &&n = 1;
    static_assert(std::is_same_v<decltype(n), int &&>);
    static_assert(std::is_same_v<decltype((n)), int &>);
  }
  {
    const int &&n = 1;
    static_assert(std::is_same_v<decltype(n), const int &&>);
    static_assert(std::is_same_v<decltype((n)), const int &>);
  }
}

void func_test() {
  static_assert(std::is_same_v<decltype(make_var<int>()), int>);
  static_assert(std::is_same_v<decltype((make_var<int>())), int>);
  static_assert(std::is_same_v<decltype(make_var<const int>()), int>);
  static_assert(std::is_same_v<decltype((make_var<const int>())), int>);

  static_assert(std::is_same_v<decltype(make_var<int &>()), int &>);
  static_assert(std::is_same_v<decltype((make_var<int &>())), int &>);
  static_assert(std::is_same_v<decltype(make_var<const int &>()), const int &>);
  static_assert(std::is_same_v<decltype((make_var<const int &>())), const int &>);

  static_assert(std::is_same_v<decltype(make_var<int &&>()), int &&>);
  static_assert(std::is_same_v<decltype((make_var<int &&>())), int &&>);
  static_assert(std::is_same_v<decltype(make_var<const int &&>()), const int &&>);
  static_assert(std::is_same_v<decltype((make_var<const int &&>())), const int &&>);
}

template <typename U, typename T> void do_test(T &&) {
  static_assert(std::is_same_v<T &&, U>);
}
void param_test() {

  {
    int n;
    do_test<int &>(n);
  }
  {
    const int n = 1;
    do_test<const int &>(n);
  }

  {
    int a;
    int &n = a;
    do_test<int &>(n);
  }
  {
    int a;
    const int &n = a;
    do_test<const int &>(n);
  }

  {
    int &&n = 1;
    do_test<int &&>(std::move(n));
  }
  {
    const int &&n = 1;
    do_test<const int &&>(std::move(n));
  }
}

int main(int argc, char *argv[]) { return 0; }
