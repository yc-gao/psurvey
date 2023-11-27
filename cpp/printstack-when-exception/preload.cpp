#include <dlfcn.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#include <exception>

#include "stacktrace.h"

extern "C" void *
__cxxabiv1::__cxa_allocate_exception(std::size_t thrown_size) noexcept {
  static void *(*func)(std::size_t) =
      (void *(*)(std::size_t))dlsym(RTLD_NEXT, "__cxa_allocate_exception");
  print_stack();
  return func(thrown_size);
}
