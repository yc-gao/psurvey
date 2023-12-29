cmake_minimum_required(VERSION 3.17)
project(cmake-template VERSION 0.0.1)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe" CACHE STRING "cxx compile flags")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -rdynamic -fno-omit-frame-pointer" CACHE STRING "cxx compile flags only debug")

include_directories(${CMAKE_CURRENT_LIST_DIR})

add_executable(demo demo.cc)

