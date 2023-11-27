#include <iostream>

#include <boost/stacktrace.hpp>

void print_stack() { std::cout << boost::stacktrace::stacktrace(); }

