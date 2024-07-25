#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("normal macros") {
  int a = 5;
  int b = 5;

  CHECK_FALSE(!(a == b));
  REQUIRE(a == b);
  CHECK_EQ(a, b);
  CHECK(doctest::Approx(0.1000001) == 0.1000002);
  CHECK(doctest::Approx(0.502) == 0.501);
}
