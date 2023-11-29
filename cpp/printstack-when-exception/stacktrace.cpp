#include <iostream>

#include <boost/stacktrace.hpp>

void print_stack() { std::cerr << boost::stacktrace::stacktrace(); }
