cmake_minimum_required(VERSION 3.17)
project(degine VERSION 0.0.1)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe" CACHE STRING "Flags used by the compiler")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -rdynamic -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-inline" CACHE STRING "Flags used by the compiler during Debug builds")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3" CACHE STRING "Flags used by the compiler during Release builds")

file(GLOB SRCS degine/*.cc)
add_library(degine ${SRCS})
target_include_directories(degine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(example)

