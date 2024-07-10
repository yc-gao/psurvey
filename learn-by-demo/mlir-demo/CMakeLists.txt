cmake_minimum_required(VERSION 3.17)
project(mlir-demo
    VERSION 0.0.1
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pipe -Wall -Wextra -Wno-unused-parameter -Wno-unused-but-set-parameter -Wno-unused-function -Wno-comment -Wno-redundant-move")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GNUInstallDirs)

find_package(MLIR REQUIRED)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_subdirectory(example)

