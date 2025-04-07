cmake_minimum_required(VERSION 3.17)
project(nv-benchmark
    VERSION 0.0.1
    LANGUAGES C CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_POSITION_INDEPENDENT_CODE True)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pipe -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CMAKE_CUDA_ARCHITECTURES 80 90)

set(proj_COMMON_INCLUDE_DIRS "${CMAKE_SOURCE_DIR};${CMAKE_BINARY_DIR}")
include_directories("${proj_COMMON_INCLUDE_DIRS}")

include(GNUInstallDirs)

add_subdirectory(3rdparty)
add_subdirectory(src)

