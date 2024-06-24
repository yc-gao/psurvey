cmake_minimum_required(VERSION 3.17)
project(cpp-demo
    VERSION 0.0.1
    LANGUAGES C CXX)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pipe -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

include(GNUInstallDirs)

# curl --create-dirs -o cmake/CPM.cmake -fsSL https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/get_cpm.cmake
set(CPM_SOURCE_CACHE "$ENV{HOME}/.cache/CPM")
include(CPM)
CPMAddPackage("gh:doctest/doctest#v2.4.11")

add_executable(HandleHolder HandleHolder.cc)
target_compile_features(HandleHolder PRIVATE cxx_std_17)
target_link_libraries(HandleHolder PRIVATE doctest::doctest)

