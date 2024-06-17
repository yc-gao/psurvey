cmake_minimum_required(VERSION 3.17)
project(cpm-demo
    VERSION 0.0.1
    LANGUAGES C CXX)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pipe -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

include(GNUInstallDirs)

set(CPM_SOURCE_CACHE "$ENV{HOME}/.cache/CPM")
include(CPM)
CPMAddPackage("gh:fmtlib/fmt#7.1.3")

add_executable(demo demo.cc)
target_compile_features(demo PRIVATE cxx_std_17)
target_link_libraries(demo PRIVATE fmt)

