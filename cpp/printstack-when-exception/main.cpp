#include "stacktrace.h"

static void /* "static" means don't export the symbol... */
myfunc2(void) {
  print_stack();
}

void myfunc(int ncalls) {
  if (ncalls > 1)
    myfunc(ncalls - 1);
  else
    myfunc2();
}

int main(int argc, char *argv[]) {
  myfunc(2);
  return 0;
}
