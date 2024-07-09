cmake_minimum_required(VERSION 3.17)
project(mlir-demo
    VERSION 0.0.1
    LANGUAGES C CXX)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pipe -Wall -Wextra -Wno-unused-parameter -Wno-unused-but-set-parameter -Wno-unused-function -Wno-comment")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

find_package(MLIR REQUIRED)
set(CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR};${CMAKE_MODULE_PATH}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

add_subdirectory(src)

