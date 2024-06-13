cmake_minimum_required(VERSION 3.17)
project(cuda-mem VERSION 0.0.1)

enable_language(CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pipe -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CMAKE_CUDA_ARCHITECTURES 80 90)

add_subdirectory(3rdparty)
add_subdirectory(src)

