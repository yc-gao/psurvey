#include <execinfo.h>

#include <iostream>

void print_stack() {
  int nptrs;
  void *buffer[200];
  char **strings;

  nptrs = backtrace(buffer, 200);
  strings = backtrace_symbols(buffer, nptrs);
  std::cerr << "dump backtrace:\n";
  if (NULL != strings) {
    for (int i = 0; i < nptrs; i++) {
      std::cerr << strings[i] << std::endl;
    }
    free(strings);
  }
}
