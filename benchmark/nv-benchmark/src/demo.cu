#include <cute/layout.hpp>

int main() {
  cute::print(
      cute::make_layout(cute::make_shape(3, 4), cute::make_stride(4, 1)));
  return 0;
}
