cmake_minimum_required(VERSION 3.17)
project(cuda-bench
    VERSION 0.0.1
    LANGUAGES C CXX CUDA)

option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pipe -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CMAKE_CUDA_ARCHITECTURES 80 90)

include(GNUInstallDirs)

set(CPM_SOURCE_CACHE "$ENV{HOME}/.cache/CPM")
include(CPM)
CPMAddPackage("gh:NVIDIA/nvbench#main")
CPMAddPackage("gh:nvidia/cccl#v2.5.0-rc1")

add_subdirectory(src)

